from datasets import load_from_disk
from transformers import AutoTokenizer

splits = load_from_disk("data/sft-splits")
train_data = splits["train"]
validation_data = splits["test"]

train_solutions = set(train_data["completion"])
validation_solutions = set(validation_data["completion"])

overlap = train_solutions & validation_solutions

print("Identical solutions in both splits:", len(overlap))

tokenizer = AutoTokenizer.from_pretrained(
    "models/qwen-base",
    local_files_only=True,
)

for split_name in ["train", "test"]:
    lengths = []

    for example in splits[split_name]:
        text = example["prompt"] + example["completion"]
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(token_ids))

    print("Split:", split_name)
    print("Longest example:", max(lengths))
    print("Examples above 512 tokens:", sum(n > 512 for n in lengths))
    print("Examples above 1024 tokens:", sum(n > 1024 for n in lengths))
