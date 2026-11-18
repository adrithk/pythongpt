# Training on a rented GPU (Vast.ai)

Written November 15, 2026; updated November 17. The Mac prepares the data and
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
2. Commit and push, so the rented machine can clone the same code.
3. Check the size of what you will upload:

~~~bash
du -sh data/s2-200k-splits
~~~

## Choosing the machine

- One GPU with 80 GB (H100 or A100 80 GB). A 0.5B model does not use
  several GPUs automatically: the planned loop uses one. Using both cards
  requires adding distributed training and measuring whether the speedup
  justifies their combined cost.
- A PyTorch template with CUDA, so PyTorch is already installed.
- **On-demand, not interruptible.** Interruptible is about half the price but
  can be taken away mid-run. If you do use it, your loop needs to be
  able to continue from its last checkpoint (step 5 of writing it).
- Disk: about 80 GB. `train_torch.py` saves weights (about 2 GB) every
  `SAVE_EVERY` = 500 updates, about 18 checkpoints over 3 epochs (about 37 GB),
  and keeps optimizer state (about 4 GB) only in the newest one (arithmetic,
  not measured).
- With a smaller card, lower `MAX_TOKENS` (padded tokens on the GPU at once):
  the output scores alone take about 0.6 MB per position, so 16,000 positions
  is sized for 80 GB.

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
~~~

~~~bash
pip install -r requirements.txt
~~~

Download the same base model revision the Mac has:

~~~bash
hf download Qwen/Qwen2.5-Coder-0.5B --revision 8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301 --local-dir models/qwen-base
~~~

Check that PyTorch sees the GPU. It should print `True` and the card's name:

~~~bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
~~~

## Copy the data up (run on the Mac)

Use the direct SSH address and port from the console. Vast.ai's own docs warn
that its proxy connection is slow for transfers over about 1 GB.

~~~bash
rsync -avP -e "ssh -p PORT" data/s2-200k-splits root@HOST:pydrithgpt/data/
~~~

## Train

~~~bash
python train_torch.py 2>&1 | tee train.log
~~~

In a second `tmux` pane, watch memory and usage:

~~~bash
watch -n 5 nvidia-smi
~~~

What to check in the first few minutes:

- The printed example counts: 197,374 training and 1,000 validation (minus
  any row the trainer finds over 1,024 tokens).
- The memory line printed after the biggest-chunk check, before training.
- `loss` is a normal number (not `nan`, not 0.0) and falls over the first
  few hundred updates.
- Time per update (the `minutes` column). Multiply by the total number of
  updates (about 9,250) to get the run time, and by the hourly price to get the
  cost. Do this before walking away.

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

Vast.ai is a marketplace, so prices move. Third-party trackers in 2026 showed
H100s at roughly USD 1.50-2.80 per hour on-demand; listings seen on November
15 were USD 2.94-3.58. The run time on an H100 has not been measured. Rough
arithmetic for S2 (77.1M tokens x 3 epochs, about 6 x 0.5B x 231M = 7e17
operations) at an assumed 25-40% of the H100's bf16 peak gives about 0.5-1 hour
of pure training; add setup, validation and saves, so plan on 1.5-3 hours of
rental. Replace this with the measured time per update.
