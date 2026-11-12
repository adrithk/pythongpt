# PC inference and HumanEval+

Run these commands in the PC's WSL project folder, using the ROCm environment
exports that worked during training. These scripts target the PC GPU through
PyTorch's CUDA interface. No retraining is needed.

## 1. Select the saved checkpoint

```bash
git pull
source .venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1
export S1_MODEL="runs/initial-sft/final"
ls "$S1_MODEL"
```

If the completed run used the recovery output, set `S1_MODEL` to
`runs/initial-sft-recovery-5500/final` instead. Use the folder that was actually
saved. It must contain the weights, configuration, and tokenizer. Git does not
transfer these ignored model files.

## 2. Try basic inference

```bash
python infer.py --model "$S1_MODEL"
```

At `Problem>`, try:

```text
Write a Python function add(a, b) that returns their sum. Return only Python code.
```

Type `/quit` to exit. Each prompt starts a fresh conversation. The model loads
once, uses greedy decoding with a 512-token answer budget, and saves responses
under `runs/inference/`. It prints generated code without executing it.

For a multiline or LeetCode problem, save the statement and required function
signature in `problem.txt`, then run:

```bash
python infer.py --model "$S1_MODEL" --prompt-file problem.txt
```

The loader checks saved parameters for NaN/Inf before moving the model to the
GPU. A finite-parameter check does not establish coding correctness. If it fails,
preserve the checkpoint and inspect the earlier numerical incident.

## 3. Generate HumanEval+ answers

Only continue after basic inference works:

```bash
python -m pip install -r requirements-eval.txt
python benchmark_humaneval.py --model "$S1_MODEL" --out runs/benchmarks/s1
```

The first benchmark run downloads the dataset. The script generates one greedy
answer per task; it does not execute the answers. `samples.jsonl` contains
extracted full solutions, `raw.jsonl` retains original responses and prompts,
and `metadata.json` records dataset identity and generation settings. Reference
solutions and hidden tests are not supplied to the model.

Use a new output directory for each run. This simple script does not resume;
after interruption preserve the partial folder and choose a new name. A
`limit=True` progress message marks a potentially truncated answer.

## 4. Grade in Docker

Docker must be running and accessible from this WSL terminal. If needed, enable
the distro's integration in Docker Desktop. Do not grade generated code directly
in your project environment.

```bash
docker version
docker pull ganler/evalplus:latest
docker image inspect ganler/evalplus:latest > runs/benchmarks/s1/grader-image.json
```

From the project root:

```bash
docker run --rm \
  --user 0:0 \
  --network none \
  --cpus 4 \
  --memory 8g \
  --pids-limit 256 \
  --cap-drop ALL \
  --cap-add DAC_OVERRIDE \
  --security-opt no-new-privileges \
  --mount "type=bind,src=$PWD/runs/benchmarks/s1,dst=/app" \
  --mount "type=bind,src=$PWD/grade_humaneval.py,dst=/grade_humaneval.py,readonly" \
  ganler/evalplus:latest \
  python /grade_humaneval.py
```

The official image includes cached benchmark data. The grader checks matching
dataset hashes, records its Python/EvalPlus versions in `grader.json`, and runs
one CPU worker. It prints original HumanEval and HumanEval+ pass@1 and saves
detailed results alongside the samples. Here "base tests" means the original
tests, not the untrained model. A score of `0.250` means 25% of problems passed.
Keep the recorded image information and reuse the same image for comparisons.

## 5. Compare with the original base

```bash
python benchmark_humaneval.py --model models/qwen-base --out runs/benchmarks/b0
```

Repeat the image-recording and grading commands with `runs/benchmarks/b0` in
place of `runs/benchmarks/s1`. Both models use the same chat format and answer
budget. Native base-completion prompting can be a separate comparison; these
settings do not automatically reproduce Qwen's published scores. Preserve all
raw outputs and report the actual settings. Training-data overlap with the
benchmark has not been audited.

Reference: [EvalPlus generation and Docker workflow](https://github.com/evalplus/evalplus#quick-start).
