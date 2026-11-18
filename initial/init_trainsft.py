import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a GPU. Run this on the rented machine.")

device = torch.device("cuda")
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", torch.cuda.get_device_name(0), f"({gpu_memory:.0f} GB)")

# ---- Recipe 2 (November 14), set up for a rented 80 GB NVIDIA GPU (November 15).
# S1 used: 512 tokens, 1 epoch, LR 2e-5, no warmup, effective batch 16, output
# runs/initial-sft. Git history has that version and the 16 GB PC version.
DATA = "data/sft-splits"  # Still the S1 data. Point this at the new mix later.
OUTPUT = "runs/sft-recipe2"  # A new folder, so S1's files are never overwritten.
MAX_LENGTH = 1024  # Prompt + completion tokens; longer examples are dropped.
EPOCHS = 3
LEARNING_RATE = 5e-5  # The peak value, reached after WARMUP_STEPS.
WARMUP_STEPS = 100
# Memory: MICROBATCH examples are on the GPU at once. The big cost is the
# output layer: 151,936 scores for every token. For 16 examples of 1,024
# tokens the estimate is about 65 GB at the worst moment, counting the
# layers' saved values as well (arithmetic, not measured).
# On a 48 GB card use 8; on a 24 GB card use 4. Training is equivalent,
# because EFFECTIVE_BATCH decides how many examples go into each update.
MICROBATCH = 16
EFFECTIVE_BATCH = 64  # Examples per weight update.
# To continue a run that was cut off, put a checkpoint folder here, for
# example "runs/sft-recipe2/checkpoint-2500". None starts from the beginning.
RESUME = None

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
    per_device_eval_batch_size=MICROBATCH,
    gradient_accumulation_steps=EFFECTIVE_BATCH // MICROBATCH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    max_length=MAX_LENGTH,
    completion_only_loss=True,
    eos_token="<|im_end|>",
    packing=False,
    # Checkpointing trades speed for memory. An 80 GB card does not need it.
    gradient_checkpointing=False,
    optim="adamw_torch",
    # The weights stay float32. bf16 runs the forward pass in 16-bit numbers,
    # which NVIDIA cards do much faster. The PC run was float32 throughout.
    fp16=False,
    bf16=True,
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

result = trainer.train(resume_from_checkpoint=RESUME)


trainer.save_metrics("train", result.metrics)

model.config.use_cache = True
final_path = f"{settings.output_dir}/final"

trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
