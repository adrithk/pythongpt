#!/usr/bin/env bash
# Benchmark rental: run on the GPU machine from the project folder, after
# copying runs/sft-torch/checkpoint-6000 and data/curated/s2-200k-dev.jsonl up.
# Generates answers only; grade them on the Mac in Docker (see CLOUD.md).
set -euo pipefail

pip install -q -r requirements.txt -r requirements-eval.txt

# Official Qwen instruct model at the same size: the reference to compare with.
hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir models/qwen-instruct
python -c "from huggingface_hub import model_info; \
print(model_info('Qwen/Qwen2.5-Coder-0.5B-Instruct').sha)" > models/qwen-instruct/REVISION

evalplus() { python benchmark_humaneval.py --dataset "$1" --model "$2" --out "runs/benchmarks/$3"; }
validation() { python benchmark_validation.py --model "$1" --out "runs/benchmarks/$2"; }

S2=runs/sft-torch/checkpoint-6000
QWEN=models/qwen-instruct

evalplus mbpp "$S2" s2-ckpt6000-mbpp
validation "$S2" s2-ckpt6000-val
evalplus humaneval "$QWEN" qwen-instruct
evalplus mbpp "$QWEN" qwen-instruct-mbpp
validation "$QWEN" qwen-instruct-val
cp models/qwen-instruct/REVISION runs/benchmarks/qwen-instruct/REVISION
echo ALLDONE
