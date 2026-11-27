"""GRPO: reinforcement learning from test results, starting from S3.

    python train_grpo.py

Every step:
  1. take TASKS_PER_STEP tasks from data/grpo/pool.jsonl (made by grpo_pool.py),
  2. let the model write GROUP answers to each, by sampling,
  3. run every answer against its task's tests: reward 1 if all pass, else 0,
  4. advantage = the answer's reward compared with its group's average,
  5. one weight update that makes above-average answers more likely and
     below-average answers less likely.

A group whose answers all pass, or all fail, has advantage 0 everywhere and
teaches nothing, so it is left out of the update.
"""

import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from curate_verify import BATCH_RUNNER, run_batch
from distill_verify import IMAGE, TIME_LIMIT, docker_command
from grade_validation import extract

# ---- Settings
BASE = "runs/sft-s3/final"
POOL = "data/grpo/pool.jsonl"
DEV = "data/grpo/dev.jsonl"
OUTPUT = Path("runs/grpo")
STEPS = 200  # Weight updates. 200 x 16 tasks = 3,200 tasks.
TASKS_PER_STEP = 16
GROUP = 8  # Answers per task.
TEMPERATURE = 0.8  # Sampling randomness. 0.8 gave 39% mixed groups for S3.
MAX_NEW_TOKENS = 512
LEARNING_RATE = 1e-6  # RL changes the model gently; SFT used 2e-5.
WARMUP_STEPS = 10
MAX_TOKENS = 16_000  # Most padded tokens on the GPU at once in the update.
GENERATE_TASKS = 16  # Tasks per generate() call (x GROUP answers at once).
# Where answers run. "docker" (the Mac): the locked-down container used for
# curation. "local": a plain process in an empty temporary folder with a
# memory and time limit, for rented GPUs, which have no Docker. Use "local"
# only on a throwaway machine: the answers are code written by the model.
SANDBOX = "docker"
TEST_BATCH = 32  # Answers per sandbox run.
TEST_WORKERS = 4  # Sandbox runs at once.
SAVE_EVERY = 50  # Steps between held-out checks and saves.
DEV_LIMIT = None  # A small number to check fewer held-out tasks (quick tests).
SEED = 0
RESUME = None  # A checkpoint folder, for example "runs/grpo/checkpoint-100".
MAX_SKIPPED = 5  # Updates with NaN/infinite gradients allowed before stopping.

# ---- Device (same as train_torch.py)
if torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.enable_cudnn_sdp(False)  # NaN gradients on H100, Nov 24.
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
USE_BF16 = device == "cuda"
print("Device:", device)

# ---- Tokenizer, tasks, model
tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
# generate() needs prompts of different lengths lined up at the RIGHT edge,
# so new tokens continue every prompt in the same column.
tokenizer.padding_side = "left"
PAD = tokenizer.pad_token_id
STOP_IDS = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def read_tasks(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


pool = read_tasks(POOL)
dev_tasks = read_tasks(DEV)[:DEV_LIMIT]
print(f"Training tasks: {len(pool)}, held-out tasks: {len(dev_tasks)}")

# The plan: which tasks each step uses. Depends only on SEED, so a resumed run
# continues with the tasks it would have seen. Wraps around if STEPS needs more.
order = list(range(len(pool)))
random.Random(SEED).shuffle(order)
plan = [
    [
        pool[order[(step * TASKS_PER_STEP + i) % len(pool)]]
        for i in range(TASKS_PER_STEP)
    ]
    for step in range(STEPS)
]

model = AutoModelForCausalLM.from_pretrained(
    RESUME or BASE, local_files_only=True, dtype=torch.float32
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
first_step = 0
if RESUME:
    state = torch.load(Path(RESUME) / "training_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    first_step = state["step"]
    print("Continuing from step", first_step)


# ---- 1-2. Write answers
@torch.inference_mode()
def generate(tasks, group, sample):
    """group answers per task. Returns one dict per answer, grouped by task."""
    if sample:
        # top_p=1 and top_k=0: sample from the model's full distribution, so
        # the probabilities computed in the update match how answers were made.
        choice = {
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": 1.0,
            "top_k": 0,
        }
    else:
        choice = {"do_sample": False}  # Greedy, as in the benchmarks.
    settings = GenerationConfig(
        **choice,
        num_return_sequences=group,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        eos_token_id=STOP_IDS,
        pad_token_id=STOP_IDS[0],
    )
    model.eval()
    answers = []
    for start in range(0, len(tasks), GENERATE_TASKS):
        part = tasks[start : start + GENERATE_TASKS]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task["instruction"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for task in part
        ]
        inputs = tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=USE_BF16):
            output = model.generate(**inputs, generation_config=settings)
        # Rows come out task by task: task 0's answers first, then task 1's...
        new_tokens = output[:, inputs.input_ids.shape[1] :].tolist()
        for row, ids in enumerate(new_tokens):
            task = row // group
            prompt_ids = inputs.input_ids[task][inputs.attention_mask[task].bool()]
            # Keep everything up to and including the first stop token; the
            # rest is padding. No stop token means the answer hit the limit.
            stop = next((i for i, t in enumerate(ids) if t in STOP_IDS), None)
            answers.append(
                {
                    "task": part[task],
                    "prompt_ids": prompt_ids.tolist(),
                    "answer_ids": ids if stop is None else ids[: stop + 1],
                    "text": tokenizer.decode(ids, skip_special_tokens=True),
                    "hit_limit": stop is None,
                }
            )
    model.train()
    return answers


# ---- 3. Run the tests
def sandbox_command():
    if SANDBOX == "docker":
        return docker_command(
            ["docker"], IMAGE, BATCH_RUNNER, TEST_BATCH * TIME_LIMIT + 60
        )
    if SANDBOX != "local":
        raise ValueError('SANDBOX must be "docker" or "local".')
    # Each answer still runs as its own process with a time limit (inside
    # BATCH_RUNNER); this adds an empty working folder and 1 GB of memory.
    setup = (
        "import os, resource, tempfile\n"
        "os.chdir(tempfile.mkdtemp())\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))\n"
    )
    return [sys.executable, "-I", "-c", setup + BATCH_RUNNER]


COMMAND = sandbox_command()


def test_batch(rows):
    passed = run_batch(COMMAND, rows)
    if passed is None:  # The whole run died: test one at a time. A crash fails.
        passed = [bool((run_batch(COMMAND, [row]) or [False])[0]) for row in rows]
    return passed


def rewards_for(answers):
    """1.0 if the answer's code passes every test of its task, else 0.0."""
    rows = [
        {"response": extract(a["text"]), "tests": a["task"]["tests"]} for a in answers
    ]
    batches = [rows[i : i + TEST_BATCH] for i in range(0, len(rows), TEST_BATCH)]
    with ThreadPoolExecutor(TEST_WORKERS) as workers:
        passed = [ok for part in workers.map(test_batch, batches) for ok in part]
    return [1.0 if ok else 0.0 for ok in passed]


# ---- 4. Advantages
def advantages_for(rewards):
    """Each reward minus its group's average, divided by the group's spread.

    All-equal groups give exactly 0 (0 / (0 + 1e-4)).
    """
    r = torch.tensor(rewards).view(-1, GROUP)  # One row per task.
    advantage = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-4)
    return advantage.view(-1).tolist()


# ---- 5. The update
def make_chunks(examples):
    """Same idea as train_torch.py: sort by length, cap padded tokens per chunk."""
    examples = sorted(examples, key=lambda e: len(e["input_ids"]))
    chunks, chunk = [], []
    for e in examples:
        if chunk and (len(chunk) + 1) * len(e["input_ids"]) > MAX_TOKENS:
            chunks.append(chunk)
            chunk = []
        chunk.append(e)
    chunks.append(chunk)
    return chunks


def chunk_loss(chunk):
    """Sum over answer tokens of -(advantage x log-probability of that token)."""
    longest = max(len(e["input_ids"]) for e in chunk)
    input_ids = torch.full((len(chunk), longest), PAD)
    labels = torch.full((len(chunk), longest), -100)
    attention_mask = torch.zeros((len(chunk), longest), dtype=torch.long)
    for row, e in enumerate(chunk):
        n = len(e["input_ids"])
        input_ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        attention_mask[row, :n] = 1
    input_ids, labels = input_ids.to(device), labels.to(device)
    advantages = torch.tensor([e["advantage"] for e in chunk], device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=USE_BF16):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits
    # Position i predicts token i + 1. Dividing by TEMPERATURE scores the
    # answers under the same distribution they were sampled from.
    scores = logits[:, :-1].float() / TEMPERATURE
    targets = labels[:, 1:]
    # cross_entropy = -log-probability of the target token (0 where ignored).
    token_log_probs = -F.cross_entropy(
        scores.reshape(-1, scores.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(targets.shape)
    # advantages[:, None] gives every token of an answer its answer's advantage.
    return -(advantages[:, None] * token_log_probs).sum()


def dev_pass_rate():
    """Share of held-out tasks the model solves with one greedy answer."""
    answers = generate(dev_tasks, 1, sample=False)
    return round(sum(rewards_for(answers)) / len(answers), 4)


def save(folder, step=None):
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    if step is not None:
        torch.save(
            {"optimizer": optimizer.state_dict(), "step": step},
            Path(folder) / "training_state.pt",
        )


def log(record):
    print(" | ".join(f"{name} {value}" for name, value in record.items()), flush=True)
    with (OUTPUT / "log.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


# ---- Training
if __name__ == "__main__":  # Importing this file (grpo_filter.py) skips training.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.train()
    if not RESUME:
        log({"step": 0, "dev_pass_rate": dev_pass_rate()})  # The starting point.

    newest_checkpoint = Path(RESUME) if RESUME else None
    skipped = 0
    for step in range(first_step, STEPS):
        started = time.time()
        learning_rate = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        torch.manual_seed(SEED * 100_000 + step)  # Same samples if resumed.
        answers = generate(plan[step], GROUP, sample=True)
        generated = time.time()
        rewards = rewards_for(answers)
        tested = time.time()
        advantages = advantages_for(rewards)

        examples = [
            {
                "input_ids": a["prompt_ids"] + a["answer_ids"],
                "labels": [-100] * len(a["prompt_ids"]) + a["answer_ids"],
                "advantage": advantage,
            }
            for a, advantage in zip(answers, advantages)
            if advantage != 0  # Groups that all passed or all failed teach nothing.
        ]
        grad_size = 0.0
        if examples:
            # Divide by all answer tokens in the update: every token counts the
            # same, however the answers were chunked.
            graded = sum(len(e["input_ids"]) - e["labels"].count(-100) for e in examples)
            for chunk in make_chunks(examples):
                (chunk_loss(chunk) / graded).backward()
            grad_size = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if math.isfinite(grad_size):
                optimizer.step()
            else:
                skipped += 1
                if skipped > MAX_SKIPPED:
                    raise FloatingPointError(f"{skipped} updates had broken gradients.")
            optimizer.zero_grad(set_to_none=True)

        group_means = torch.tensor(rewards).view(-1, GROUP).mean(dim=1)
        done = step + 1
        log(
            {
                "step": f"{done}/{STEPS}",
                "reward": round(sum(rewards) / len(rewards), 3),
                "mixed": int(((group_means > 0) & (group_means < 1)).sum()),
                "all_pass": int((group_means == 1).sum()),
                "all_fail": int((group_means == 0).sum()),
                "tokens": round(sum(len(a["answer_ids"]) for a in answers) / len(answers)),
                "hit_limit": sum(a["hit_limit"] for a in answers),
                "grad": round(grad_size, 3),
                "skipped": skipped,
                "lr": f"{learning_rate:.1e}",
                "seconds": f"{generated - started:.0f}+{tested - generated:.0f}"
                f"+{time.time() - tested:.0f}",
            }
        )

        if done % SAVE_EVERY == 0 or done == STEPS:
            log({"step": done, "dev_pass_rate": dev_pass_rate()})
            folder = OUTPUT / (f"checkpoint-{done}" if done < STEPS else "final")
            save(folder, step=done)
            # Only the newest checkpoint keeps the large optimizer state.
            if newest_checkpoint and newest_checkpoint != folder:
                (newest_checkpoint / "training_state.pt").unlink(missing_ok=True)
            newest_checkpoint = folder

    print("Saved the trained model to", OUTPUT / "final")
