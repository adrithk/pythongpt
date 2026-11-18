from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer


coding = load_dataset(
    "OpenCoder-LLM/opc-sft-stage2",
    "educational_instruct",
    split="train",
)
print("Total examples:", len(coding))

tokenizer = AutoTokenizer.from_pretrained(
    "models/qwen-base",
    local_files_only=True,
)

tokenizer.eos_token = "<|im_end|>"


def format_example(example):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["instruction"]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    completion = example["code"] + tokenizer.eos_token

    return {
        "prompt": prompt,
        "completion": completion,
    }


prepared = coding.map(
    format_example,
    remove_columns=coding.column_names,
)

print("Prepared examples:", len(prepared))
print("Prepared columns:", prepared.column_names)

prepared.save_to_disk("data/prepared-sft")

splits = prepared.train_test_split(
    test_size=0.01,
    seed=42,
)

train_data = splits["train"]
validation_data = splits["test"]

train_solutions = set(train_data["completion"])

move_indices = []
keep_indices = []

for index, solution in enumerate(validation_data["completion"]):
    if solution in train_solutions:
        move_indices.append(index)
    else:
        keep_indices.append(index)

moved_examples = validation_data.select(move_indices)

train_data = concatenate_datasets([train_data, moved_examples])
validation_data = validation_data.select(keep_indices)

splits["train"] = train_data
splits["test"] = validation_data

print("Moved into training:", len(moved_examples))
print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))

splits.save_to_disk("data/sft-splits")
