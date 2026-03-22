from llguidance import LLMatcher, LLTokenizer, LLParserLimits
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import torch
from llguidance import TokenizerWrapper

from dgrammar.grammar_cache import get_cached_grammar

model_name_map = {
    "LLaDA": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream": "Qwen/Qwen2.5-7B",
}

# Module-level cache: building LLTokenizer is expensive (full vocab scan).
# Key: resolved model_name string (or None), Value: LLTokenizer instance.
_tokenizer_cache: dict = {}


def _get_cached_tokenizer(model_name: str | None) -> LLTokenizer:
    key = model_name
    if key not in _tokenizer_cache:
        if model_name is None:
            _tokenizer_cache[key] = LLTokenizer("byte")
        else:
            resolved = model_name_map.get(model_name, model_name)
            tokenizer_path = hf_hub_download(
                repo_id=resolved,
                filename="tokenizer.json",
            )
            _tokenizer_cache[key] = LLTokenizer(tokenizer_path)
    return _tokenizer_cache[key]


class Checker():
    tokenizer: LLTokenizer
    matcher: LLMatcher
    tokens: list[int]

    def __init__(self, grammar: str, model_name=None) -> None:
        self.tokenizer = _get_cached_tokenizer(model_name)
        grm = get_cached_grammar(grammar)
        self._limits = LLParserLimits(max_items_in_row=20000, step_max_items=600000)
        self.matcher = LLMatcher(self.tokenizer, grm, log_level=1, limits=self._limits)
        self.tokens = []
        self._grammar = grammar
        self._grm = grm
        self._model_name = model_name

    def clone(self) -> "Checker":
        """Fresh matcher: shared tokenizer + shared compiled grammar; new LLMatcher only."""
        c = object.__new__(Checker)
        c.tokenizer = self.tokenizer
        c._grammar = self._grammar
        c._grm = self._grm
        c._model_name = self._model_name
        c._limits = self._limits
        c.tokens = []
        c.matcher = LLMatcher(c.tokenizer, self._grm, log_level=1, limits=self._limits)
        return c

    def __deepcopy__(self, memo):
        """Stateful copy using LLMatcher.deep_copy() — O(Rust state size), no token replay."""
        c = object.__new__(Checker)
        c.tokenizer = self.tokenizer
        c._grammar = self._grammar
        c._grm = self._grm
        c._model_name = self._model_name
        c._limits = self._limits
        c.tokens = list(self.tokens)
        c.matcher = self.matcher.deep_copy()
        return c

    def validate_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)        
        return self.matcher.validate_tokens(next_tokens) == len(next_tokens)

    def validate_tokens(self, next_tokens):
        assert not self.matcher.is_error(), "Matcher is in error state"
        if next_tokens is None or len(next_tokens) == 0:
            return True
        if self.matcher.is_stopped():
            return False
        res = self.matcher.validate_tokens(next_tokens) == len(next_tokens)
        if self.matcher.is_error():
            print(f"Cannot validate tokens: {self.tokenizer.dbg_tokens(next_tokens)}")
            print(next_tokens)
            print(self.matcher.validate_tokens(next_tokens))
            print("Error state:")
            print(self.matcher.get_error())
            print("-------------------")
            assert False
        return res

    def end_with_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)        
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            self.matcher.rollback(count)
            return False
        if self.matcher.is_accepting():
            self.matcher.rollback(count)
            return True
        else:
            self.matcher.rollback(count)
            return False

    def end_with_tokens(self, next_tokens):
        if next_tokens is None or len(next_tokens) == 0:
            return self.matcher.is_accepting()
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            self.matcher.rollback(count)
            return False
        if self.matcher.is_accepting():
            self.matcher.rollback(count)
            return True
        else:
            self.matcher.rollback(count)
            return False
        
    def reset(self):
        self.matcher.reset()

    def consume_str(self, next_str):
        next_tokens = self.tokenizer.tokenize_str(next_str)
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            print(f"Cannot consume {next_str}")
            self.matcher.rollback(count)
            return False
        return True

    def consume_tokens(self, next_tokens):
        assert not self.matcher.is_error(), "Matcher is in error state"
        if next_tokens is None or len(next_tokens) == 0:
            return True
        count = self.matcher.try_consume_tokens(next_tokens)
        if count != len(next_tokens):
            print(f"Cannot consume {next_tokens}")
            next_str = self.tokenizer.dbg_tokens(next_tokens)
            print(f"count: {count}, len: {len(next_tokens)}")
            print(f"Cannot consume {next_str}")
            self.matcher.rollback(count)
            return False
        assert not self.matcher.is_error()
        self.tokens.extend(next_tokens)
        return True

    def dbg_tokens(self, tokens):
        res =  self.tokenizer.dbg_tokens(tokens)
        # assert not self.matcher.is_error()
        return res

    def compute_mask(self) -> torch.Tensor:
        """
        Compute the token mask, with one byte per tokenizer word, for the next parsing step.
        Entries are either 0 (not allowed) or 1 (allowed).
        """
        assert not self.matcher.is_error(), "Cannot compute mask in error state"
        masks = self.matcher.compute_logit_bias()
        masks = torch.tensor([1 if m > 0 else 0 for m in masks], dtype=torch.float32)
        assert not self.matcher.is_error(), f"Error occurred during mask computation: {self.matcher.get_error()}, tokens: {self.tokens}"
        return masks
    
    def rollback(self, count: int):
        assert not self.matcher.is_error(), "Cannot rollback in error state"
        if count <= 0:
            return True
        res = self.matcher.rollback(count)
        # assert res
        assert not self.matcher.is_error()
        self.tokens = self.tokens[:-count]
        return res

    def is_stoped(self) -> bool:
        # assert not self.matcher.is_error()
        return self.matcher.is_stopped()

    def is_accepting(self) -> bool:
        # assert not self.matcher.is_error()
        return self.matcher.is_accepting()

    def is_error(self) -> bool:
        return self.matcher.is_error()