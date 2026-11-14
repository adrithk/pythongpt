import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the PC GPU.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("ROCm version:", torch.version.hip)

# ---- Recipe 2 (November 14). S1 used: 512 tokens, 1 epoch, LR 2e-5, no warmup,
# effective batch 16, output runs/initial-sft. Git history has that version.
DATA = "data/sft-splits"  # Still the S1 data. Point this at the new mix later.
OUTPUT = "runs/sft-recipe2"  # A new folder, so S1's files are never overwritten.
MAX_LENGTH = 1024  # Prompt + completion tokens; longer examples are dropped.
EPOCHS = 3
LEARNING_RATE = 5e-5  # The peak value, reached after WARMUP_STEPS.
WARMUP_STEPS = 100
# Memory: MICROBATCH examples are on the GPU at once. Weights, gradients and
# optimizer state already use about 8 GB in float32, so keep this small.
# Raise it to 2 only after watching GPU memory on a short run.
MICROBATCH = 1
EFFECTIVE_BATCH = 64  # Examples per weight update.

splits = load_from_disk(DATA)

tokenizer = AutoTokenizer.from_pretrained(
    "models/qwen-base",
    local_files_only=True,
)
tokenizer.eos_token = "<|im_end|>"


def fits_length(example):
    text = example["prompt"] + example["completion"]
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return len(token_ids) <= MAX_LENGTH


filtered = splits.filter(fits_length)

train_data = filtered["train"]
validation_data = filtered["test"]

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))

model = AutoModelForCausalLM.from_pretrained(
    "models/qwen-base",
    local_files_only=True,
    dtype=torch.float32,
    attn_implementation="sdpa",
)

model = model.to(device)
model.config.use_cache = False

print("Model device:", next(model.parameters()).device)


settings = SFTConfig(
    output_dir=OUTPUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=MICROBATCH,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=EFFECTIVE_BATCH // MICROBATCH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    max_length=MAX_LENGTH,
    completion_only_loss=True,
    eos_token="<|im_end|>",
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    logging_steps=10,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=250,  # One step now covers 64 examples, not 16.
    save_steps=250,
    save_total_limit=6,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=settings,
    train_dataset=train_data,
    eval_dataset=validation_data,
    processing_class=tokenizer,
)

result = trainer.train()


trainer.save_metrics("train", result.metrics)

model.config.use_cache = True
final_path = f"{settings.output_dir}/final"

trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
