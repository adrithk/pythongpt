# Training on a rented GPU (Vast.ai)

Written November 15, 2026. The Mac prepares the data and
holds the code; a rented NVIDIA GPU does the training with `train_torch.py`,
the plain-PyTorch loop. It reads `data/s2-200k-splits` (197,374 training rows,
1,000 validation rows) and writes to `runs/sft-torch`.

Status: **nothing here has been run on a rented machine.** On November 17 a copy
of the saved `train_torch.py` ran end to end on the Mac against the S2 data with
a tiny stand-in model and a smaller chunk size (training, validation,
checkpoints, final save). CUDA, bf16, real-model memory and speed are untested,
so treat the first minutes of the first rental as the real test.

## What lives where

| Thing | In Git? | How it reaches the rented machine |
| --- | --- | --- |
| Code | yes | `git clone` |
| Base model (`models/qwen-base`) | no | download from Hugging Face there |
| Training data (`data/s2-200k-splits`, about 300 MB) | no | copy from the Mac |
| Results (`runs/`) | no | copy back to the Mac **before destroying** |

A rented machine is deleted when you destroy it. Anything you did not copy back
is gone.

## Before you rent (on the Mac)

1. Build the training set (`CURATE.md`) and set `DATA` in `train_torch.py`
   (done: `data/s2-200k-splits`).
2. The trainer is already on `origin/main` at `15a528f`. Record the revision
   and any training settings you change on the rental.
3. Check the size of what you will upload:

~~~bash
du -sh data/s2-200k-splits
~~~

## Choosing the machine

- Start with one **A100 80 GB** if it is substantially cheaper than an H100.
  Pick an H100 SXM 80 GB if shorter wall time matters more or the A100's
  measured time is too long. Start the A100 at `MAX_TOKENS = 12_000`.
  An RTX 5090 32 GB needs smaller chunks (try
  `MAX_TOKENS = 4_000` first) and a Blackwell-compatible PyTorch image.
  The trainer does not use
  several GPUs automatically: the planned loop uses one. Using both cards
  requires adding distributed training and measuring whether the speedup
  justifies their combined cost.
- A PyTorch template with CUDA, so PyTorch is already installed.
- Use an on-demand offer from a verified host. Check historical reliability,
  preferably 99% or higher, and the host's bandwidth fees.
- Disk: about 80 GB. `train_torch.py` saves weights (about 2 GB) every
  `SAVE_EVERY` = 500 updates, about 18 checkpoints over 3 epochs (about 37 GB),
  and keeps optimizer state (about 4 GB) only in the newest one (arithmetic,
  not measured).
- Treat `MAX_TOKENS` as a padded-position cap, not a VRAM guarantee. At
  16,000 positions, one float32 tensor of output scores is about 9.7 GB,
  before other loss buffers and model activations.

## Set up the machine

Vast.ai's console shows the SSH command for your instance. Once connected,
start `tmux` first, so training survives if your connection drops:

~~~bash
tmux new -s train
~~~

Get the code and the libraries. Do not reinstall PyTorch; the template's build
already matches the GPU.

~~~bash
git clone <your repository URL> pydrithgpt && cd pydrithgpt
git checkout --detach 15a528f
mkdir -p data
~~~

~~~bash
python -m pip install -r requirements.txt
~~~

Download the same base model revision the Mac has:

~~~bash
hf download Qwen/Qwen2.5-Coder-0.5B --revision 8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301 --local-dir models/qwen-base
~~~

Check that PyTorch can run a bf16 kernel on the GPU:

~~~bash
python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); x = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16); print(torch.cuda.get_device_name(0), torch.version.cuda, (x @ x)[0, 0].item())"
~~~

## Copy the data up (run on the Mac)

Use the direct SSH address and port from the console. Vast.ai's own docs warn
that its proxy connection is slow for transfers over about 1 GB.

~~~bash
rsync -avP -e "ssh -p PORT" data/s2-200k-splits root@HOST:pydrithgpt/data/
~~~

## Train

~~~bash
set -o pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_torch.py 2>&1 | tee train.log
~~~

The setting in front lets PyTorch grow GPU memory blocks instead of failing and
retrying when memory is fragmented (the first H100 run printed many "memory
allocation failed with OOM" warnings without it).

In a second `tmux` pane, watch memory and usage:

~~~bash
watch -n 5 nvidia-smi
~~~

What to check in the first few minutes:

- The printed example counts: 197,374 training and 1,000 validation (minus
  any row the trainer finds over 1,024 tokens).
- Check GPU memory after the first optimizer step. The startup probe uses the
  64 longest rows as a stress sample and runs before AdamW creates its state.
  The actual training plan can produce a larger chunk.
- Check that loss stays finite, then track validation loss across checkpoints.
- Estimate steady-state seconds per update from the timestamps at updates
  20 and 100. Multiply by 9,252 and divide by 3,600 to estimate training
  hours. Add setup, validation, saving, and transfer time to budget the rental.
- For an early recovery point on the full run, change the checkpoint condition
  to `if done < total_steps and (done == 100 or done % SAVE_EVERY == 0):`.
  Leave `LIMIT = None` for this timing run. A resume checkpoint needs
  `training_state.pt`. The trainer keeps it only in the latest checkpoint.

If it stops with an out-of-memory error, halve `MAX_TOKENS` and start again
(or set `RESUME` to the newest checkpoint). Each update stays 64 rows; only how
they are split into chunks changes.

## Benchmark, copy back, destroy

Generation (`infer.py`, `benchmark_humaneval.py`) can run on the rented GPU.
Grading needs Docker, and a Vast.ai instance is itself a container, so copy the
generated samples back and grade on the Mac.

Copy results to the Mac (run on the Mac). The `final` folder is the trained
model; `log.jsonl` and `train.log` record the run:

~~~bash
rsync -avP -e "ssh -p PORT" root@HOST:pydrithgpt/runs/sft-torch/final runs/sft-torch/
~~~

~~~bash
rsync -avP -e "ssh -p PORT" root@HOST:pydrithgpt/runs/sft-torch/log.jsonl runs/sft-torch/
~~~

~~~bash
rsync -avP -e "ssh -p PORT" root@HOST:pydrithgpt/train.log runs/sft-torch/
~~~

Copy any `checkpoint-...` folders you want to compare as well. Then **destroy**
the instance in the console. A stopped instance still charges for its disk.

## Cost

At USD 0.65/hour for an A100 and USD 1.75/hour for an H100, the H100 would
need to be about 2.7 times as fast to save money on training time. Check
the actual offers with the desired disk allocation. The S2 trainer's
throughput and memory use on these GPUs have yet to be measured.
[Vast pricing documentation](https://docs.vast.ai/guides/instances/pricing)

## Benchmark-only rental

For measuring models, not training: about 30 minutes of an H100, 50 GB disk.
After cloning (no data set or base model needed), copy up the model and the
validation rows from the Mac:

~~~bash
rsync -avP -e "ssh -p PORT" runs/sft-torch/checkpoint-6000 root@HOST:/workspace/pythongpt/runs/sft-torch/
~~~

~~~bash
rsync -avP -e "ssh -p PORT" data/curated/s2-200k-dev.jsonl root@HOST:/workspace/pythongpt/data/curated/
~~~

Then on the GPU machine, inside tmux:

~~~bash
./run_benchmarks.sh 2>&1 | tee bench.log
~~~

It downloads Qwen2.5-Coder-0.5B-Instruct as a reference and generates
MBPP+, HumanEval+ and validation-set answers into `runs/benchmarks/`. Copy that
folder back, destroy the instance, then grade on the Mac: EvalPlus folders with
the Docker command in INFERENCE.md (`grade_humaneval.py` reads the data set name
from metadata.json), validation folders with `python grade_validation.py <folder>`.
