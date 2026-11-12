import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
)
from evalplus.sanitize import sanitize

from infer import generate, load_model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=False,
    )

    problems = get_human_eval_plus()

    tokenizer, model = load_model(args.model)

    metadata = {
        "model": str(Path(args.model).resolve()),
        "dataset_hash": get_human_eval_plus_hash(),
        "task_count": len(problems),
        "packages": {
            package: version(package)
            for package in (
                "torch",
                "transformers",
                "evalplus",
            )
        },
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "attention": "sdpa",
        "do_sample": False,
        "max_new_tokens": 512,
        "prompt_mode": (
            "Qwen chat; full function; no assistant prefill"
        ),
        "extraction": (
            "evalplus.sanitize.sanitize with required entry point"
        ),
    }

    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    instruction_prefix = (
        "Implement the following Python function. "
        "Keep its name and signature. "
        "Return the complete Python code, "
        "including any needed imports.\n\n"
    )

    samples_path = args.out / "samples.jsonl"
    raw_path = args.out / "raw.jsonl"

    with (
        samples_path.open("x", encoding="utf-8") as samples,
        raw_path.open("x", encoding="utf-8") as raw,
    ):
        for index, (task_id, task) in enumerate(
            problems.items(),
            start=1,
        ):
            instruction = instruction_prefix + task["prompt"]

            answer, hit_limit = generate(
                tokenizer,
                model,
                instruction,
            )

            solution = sanitize(
                answer,
                entrypoint=task["entry_point"],
            )

            sample_record = {
                "task_id": task_id,
                "solution": solution,
            }

            raw_record = {
                "task_id": task_id,
                "prompt": instruction,
                "answer": answer,
                "hit_limit": hit_limit,
            }

            samples.write(
                json.dumps(sample_record) + "\n"
            )
            raw.write(
                json.dumps(raw_record) + "\n"
            )

            samples.flush()
            raw.flush()

            print(
                f"{index}/{len(problems)} "
                f"{task_id} limit={hit_limit}",
                flush=True,
            )

    print(
        "Generation complete. "
        "Grade samples.jsonl separately inside Docker."
    )


if __name__ == "__main__":
    main()
