#!/bin/bash
# Run all benchmarks aligned with LAVE paper Table 1
# Model: LLaDA-8B-Instruct
# Datasets: jsonschema, THUDM/humaneval-x/cpp, smiles
# Modes: unconstrained (No-CD), constrained (IG-CD/ETH)
# Settings: L=256, T=128, temp=0.2, 10 seeds

set -e

cd /home/gmchiu/Documents/GitHub/dgrammar
source .venv/bin/activate

MODEL="GSAI-ML/LLaDA-8B-Instruct"
STEPS=128
MAX_TOKENS=256
TEMP=0.2
TIMEOUT=600

mkdir -p results

for DATASET in jsonschema "THUDM/humaneval-x/cpp" smiles; do
    DS_SAFE=$(echo "$DATASET" | tr '/' '_')

    for SEED in 0 1 2 3 4 5 6 7 8 9; do
        # No-CD (unconstrained)
        OUT="results/${DS_SAFE}_nocd_s${SEED}.jsonl"
        if [ ! -f "$OUT" ]; then
            echo "=== No-CD: $DATASET seed=$SEED ==="
            python -m constrained_diffusion.eval.dllm.generic_inference \
                --model_name "$MODEL" \
                --dataset-name "$DATASET" \
                --constrained False \
                --steps $STEPS \
                --max-tokens $MAX_TOKENS \
                --temp $TEMP \
                --seed $SEED \
                --timeout $TIMEOUT \
                --output_file "$OUT"
        fi

        # IG-CD (ETH constrained)
        OUT="results/${DS_SAFE}_igcd_s${SEED}.jsonl"
        if [ ! -f "$OUT" ]; then
            echo "=== IG-CD: $DATASET seed=$SEED ==="
            python -m constrained_diffusion.eval.dllm.generic_inference \
                --model_name "$MODEL" \
                --dataset-name "$DATASET" \
                --constrained True \
                --steps $STEPS \
                --max-tokens $MAX_TOKENS \
                --temp $TEMP \
                --seed $SEED \
                --timeout $TIMEOUT \
                --output_file "$OUT"
        fi
    done
done

echo "All benchmarks done. Run eval with: bash eval/check_all_individually.sh results/*.jsonl"
