"""Have a model solve the S2 validation problems; grade with grade_validation.py.

The 1,000 validation rows were held out of training and each has tests, so the
share solved measures skill on problems like the training data. Comparing it
with HumanEval+/MBPP+ separates "cannot solve this kind of problem" from
"cannot handle the benchmark's format".

    python benchmark_validation.py --model runs/sft-torch/checkpoint-6000 --out runs/benchmarks/s2-ckpt6000-val

Answers are generated in batches (left-padded, greedy, 512 new tokens), which
is much faster than one at a time; batching can change a few greedy answers
slightly, so compare models with the same script, not with infer.py output.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import GenerationConfig

from infer import device_name, load_model, pick_device

VALIDATION = Path("data/curated/s2-200k-dev.jsonl")
MAX_NEW_TOKENS = 512


@torch.inference_mode()
def generate_batch(tokenizer, model, instructions):
    """Greedy answers for several instructions at once. Returns (text, hit_limit) pairs."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in instructions
    ]
    tokenizer.padding_side = "left"  # New tokens must follow each prompt directly.
    inputs = tokenizer(
        prompts, add_special_tokens=False, padding=True, return_tensors="pt"
    ).to(model.device)
    stop_ids = [
        tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
    ]
    settings = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    output = model.generate(**inputs, generation_config=settings)
    results = []
    for row in output[:, inputs.input_ids.shape[1] :]:
        ids = row.tolist()
        ended = [i for i, token in enumerate(ids) if token in stop_ids]
        ids = ids[: ended[0]] if ended else ids
        results.append((tokenizer.decode(ids, skip_special_tokens=True), not ended))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=VALIDATION)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N rows")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    rows = [row for row in rows if row["tests"]]
    if args.limit:
        rows = rows[: args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer, model = load_model(args.model)
    metadata = {
        "model": str(Path(args.model).resolve()),
        "data": str(args.data),
        "rows": len(rows),
        "gpu": device_name(pick_device()),
        "dtype": "float32",
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch": args.batch,
        "prompt": "the row's instruction as the only user message (training format)",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Similar lengths together means less padding per batch.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["instruction"]))
    with (args.out / "answers.jsonl").open("x", encoding="utf-8") as file:
        for start in range(0, len(order), args.batch):
            batch = [rows[i] for i in order[start : start + args.batch]]
            answers = generate_batch(
                tokenizer, model, [row["instruction"] for row in batch]
            )
            for row, (answer, hit_limit) in zip(batch, answers):
                record = {
                    "id": row["id"],
                    "source": row["source"],
                    "answer": answer,
                    "hit_limit": hit_limit,
                    "tests": row["tests"],
                }
                file.write(json.dumps(record) + "\n")
            file.flush()
            print(f"{min(start + args.batch, len(order))}/{len(order)}", flush=True)
    print("Done. Grade with grade_validation.py on a machine with Docker.")


if __name__ == "__main__":
    main()
