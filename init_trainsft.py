import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

device = torch.device("mps")
MAX_LENGTH = 512

splits = load_from_disk("data/sft-splits")

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

device = torch.device("mps")

model = AutoModelForCausalLM.from_pretrained(
    "models/qwen-base",
    local_files_only=True,
    torch_dtype=torch.float32,
    attn_implementation="sdpa",
)

model = model.to(device)
model.config.use_cache = False

print("Model device:", next(model.parameters()).device)


settings = SFTConfig(
    output_dir="runs/initial-sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
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
    logging_steps=1,
    save_steps=500,
    save_total_limit=2,
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
