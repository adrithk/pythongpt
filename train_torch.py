import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Settings
# S3 (option B): new KodCode and loosened-filter OpenCodeInstruct rows plus a
# 15% replay of S2, continuing from S2's best checkpoint. S2 used
# DATA "data/s2-200k-splits", BASE "models/qwen-base", OUTPUT runs/sft-torch,
# EPOCHS 3 (git 2423a06).
DATA = "data/s3-splits"
BASE = "runs/sft-torch/checkpoint-6000"
OUTPUT = Path("runs/sft-s3")
MAX_LENGTH = 1024  # Prompt + completion tokens; longer rows are dropped.
EPOCHS = 2  # S2's third epoch raised validation loss.
# The peak value, reached after WARMUP_STEPS updates. OpenCoder used 5e-5 at
# batch 512; 2e-5 is about that scaled by sqrt(64 / 512) for AdamW.
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EFFECTIVE_BATCH = 64  # Rows per weight update.
MAX_TOKENS = 16_000  # Most padded tokens on the GPU at once (rows x longest row).
LOG_EVERY = 10  # Updates between printed lines.
SAVE_EVERY = 500  # Updates between validation checks and saves.
SEED = 42
RESUME = None  # A checkpoint folder, for example "runs/sft-torch/checkpoint-2500".
LIMIT = None  # A small number such as 200 to try the script on a few rows.
MAX_SKIPPED = 20  # Updates with NaN/infinite gradients allowed before stopping.

# ---- Device: the rented GPU if there is one, otherwise the Mac
if torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    # PyTorch's cuDNN attention kernel returned NaN gradients for some padded
    # chunks on an H100 (Nov 24, 2026: 5 of the first 324 updates, at both
    # LR 5e-5 and 2e-5). The other attention kernels gave finite gradients on
    # the same chunks, so turn cuDNN attention off.
    torch.backends.cuda.enable_cudnn_sdp(False)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
USE_BF16 = device == "cuda"  # 16-bit forward pass: much faster on NVIDIA cards.
print("Device:", device)

# ---- Data
tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
tokenizer.eos_token = "<|im_end|>"
PAD = tokenizer.pad_token_id
splits = load_from_disk(DATA)


def encode(example):
    prompt_ids = tokenizer(example["prompt"])["input_ids"]
    all_ids = tokenizer(example["prompt"] + example["completion"])["input_ids"]
    labels = [-100] * len(prompt_ids) + all_ids[len(prompt_ids) :]
    return {"input_ids": all_ids, "labels": labels}


def collate(examples):
    longest = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), longest), PAD)
    labels = torch.full((len(examples), longest), -100)
    attention_mask = torch.zeros((len(examples), longest), dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_rows(split):
    """Encode every row of one split and drop the ones that are too long."""
    rows = splits[split]
    if LIMIT:
        rows = rows.select(range(min(LIMIT, len(rows))))
    encoded = [encode(row) for row in rows]
    return [e for e in encoded if len(e["input_ids"]) <= MAX_LENGTH]


def graded_tokens(examples):
    """How many tokens the model is graded on (labels that are not -100)."""
    return sum(len(e["labels"]) - e["labels"].count(-100) for e in examples)


def make_chunks(examples):
    """Split rows into chunks that fit on the GPU.

    A chunk's size on the GPU is rows x longest row, because short rows are
    padded. Sorting by length first keeps similar rows together.
    """
    examples = sorted(examples, key=lambda e: len(e["input_ids"]))
    chunks, chunk = [], []
    for e in examples:  # Shortest first, so the newest row is the longest.
        if chunk and (len(chunk) + 1) * len(e["input_ids"]) > MAX_TOKENS:
            chunks.append(chunk)
            chunk = []
        chunk.append(e)
    chunks.append(chunk)
    return chunks


print("Encoding the data...")
train_rows = load_rows("train")
validation_rows = load_rows("test")
print("Training examples:", len(train_rows))
print("Validation examples:", len(validation_rows))

# ---- The plan: which rows go into each update, for the whole run
# Every epoch shuffles the rows differently, then cuts them into groups of
# EFFECTIVE_BATCH. The plan depends only on SEED, so a resumed run continues
# with exactly the rows it would have seen.
plan = []
for epoch in range(EPOCHS):
    order = list(range(len(train_rows)))
    random.Random(SEED + epoch).shuffle(order)
    for start in range(0, len(order), EFFECTIVE_BATCH):
        plan.append(order[start : start + EFFECTIVE_BATCH])
total_steps = len(plan)
print("Weight updates in this run:", total_steps)

# ---- Model and optimizer
model = AutoModelForCausalLM.from_pretrained(
    RESUME or BASE, local_files_only=True, dtype=torch.float32
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

first_step = 0
if RESUME:
    state = torch.load(Path(RESUME) / "training_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    first_step = state["step"]
    print("Continuing from update", first_step)


def learning_rate_at(step):
    """Climb in a straight line during warmup, then follow a cosine curve to 0."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def chunk_loss(chunk):
    """Add up the loss over every graded token in one chunk."""
    batch = {name: tensor.to(device) for name, tensor in collate(chunk).items()}
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=USE_BF16):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    # The scores at position i predict the token at position i + 1, so drop
    # the last position's scores and the first position's label.
    scores = logits[:, :-1].float()
    targets = batch["labels"][:, 1:]
    return F.cross_entropy(
        scores.reshape(-1, scores.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,  # Skips the prompt and the padding.
        reduction="sum",
    )


@torch.no_grad()
def validation_loss():
    """Average loss per graded token on rows the model never trains on."""
    model.eval()
    total = sum(chunk_loss(chunk).item() for chunk in make_chunks(validation_rows))
    model.train()
    return total / graded_tokens(validation_rows)


def save(folder, step=None):
    """Save the weights. With a step, also save what is needed to continue."""
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


OUTPUT.mkdir(parents=True, exist_ok=True)
model.train()

# ---- Memory check: run the biggest chunk first, before the long run starts
longest_rows = sorted(train_rows, key=lambda e: len(e["input_ids"]))[-EFFECTIVE_BATCH:]
biggest = max(make_chunks(longest_rows), key=lambda c: len(c) * len(c[-1]["input_ids"]))
chunk_loss(biggest).backward()
optimizer.zero_grad(set_to_none=True)
if device == "cuda":
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Biggest chunk fits. Peak GPU memory: {peak:.1f} GB")

# ---- Training
newest_checkpoint = Path(RESUME) if RESUME else None
started = time.time()
running_loss = 0.0
biggest_grad = 0.0  # Largest gradient size since the last printed line.
skipped = 0

for step in range(first_step, total_steps):
    # Set this update's learning rate.
    learning_rate = learning_rate_at(step)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    # One update uses EFFECTIVE_BATCH rows, fed to the GPU a chunk at a time.
    # Dividing by the graded tokens of the whole group makes the result the
    # average loss per token, however the rows were chunked.
    rows = [train_rows[i] for i in plan[step]]
    graded = graded_tokens(rows)
    loss_value = 0.0
    for chunk in make_chunks(rows):
        loss = chunk_loss(chunk) / graded
        loss.backward()  # Gradients add up across chunks.
        loss_value += loss.item()

    # Stop before touching the weights if anything went wrong.
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"Loss became NaN or infinity at update {step + 1}.")
    # clip_grad_norm_ returns the gradients' total size before clipping.
    grad_size = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

    done = step + 1
    if math.isfinite(grad_size):
        # Apply the gradients.
        optimizer.step()
        biggest_grad = max(biggest_grad, grad_size)
    else:
        # A NaN or infinite gradient would ruin the weights: skip this update
        # (its 64 rows are lost), and stop if it keeps happening.
        skipped += 1
        log({"update": f"{done}/{total_steps}", "skipped": "NaN/inf gradients"})
        if skipped > MAX_SKIPPED:
            raise FloatingPointError(f"{skipped} updates had broken gradients.")
    optimizer.zero_grad(set_to_none=True)  # Clear them for the next update.

    running_loss += loss_value
    if done % LOG_EVERY == 0 or done == total_steps:
        count = (done - 1) % LOG_EVERY + 1  # Updates since the last printed line.
        log(
            {
                "update": f"{done}/{total_steps}",
                "loss": round(running_loss / count, 4),
                "lr": f"{learning_rate:.2e}",
                "grad": round(biggest_grad, 2),
                "skipped": skipped,
                "minutes": round((time.time() - started) / 60, 1),
            }
        )
        running_loss = 0.0
        biggest_grad = 0.0

    if done % SAVE_EVERY == 0 and done < total_steps:
        log(
            {
                "update": f"{done}/{total_steps}",
                "validation_loss": round(validation_loss(), 4),
            }
        )
        folder = OUTPUT / f"checkpoint-{done}"
        save(folder, step=done)
        # Optimizer state is large and only needed to continue a run, so only
        # the newest checkpoint keeps it. Older ones keep their weights.
        if newest_checkpoint and newest_checkpoint != folder:
            (newest_checkpoint / "training_state.pt").unlink(missing_ok=True)
        newest_checkpoint = folder

# ---- Finish
log(
    {
        "update": f"{total_steps}/{total_steps}",
        "validation_loss": round(validation_loss(), 4),
    }
)
save(OUTPUT / "final")
print("Saved the trained model to", OUTPUT / "final")
