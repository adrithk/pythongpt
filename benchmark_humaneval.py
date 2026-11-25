"""Generate answers for HumanEval+ or MBPP+ (EvalPlus). Grading happens separately.

    python benchmark_humaneval.py --model runs/sft-torch/final --out runs/benchmarks/s2
    python benchmark_humaneval.py --dataset mbpp --model ... --out runs/benchmarks/s2-mbpp

Grade the output folder with grade_humaneval.py inside the EvalPlus Docker image.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.sanitize import sanitize

from infer import device_name, generate, load_model, pick_device

DATASETS = {
    # name -> (tasks, dataset hash, text placed before each task's prompt)
    "humaneval": (
        get_human_eval_plus,
        get_human_eval_plus_hash,
        (
            "Implement the following Python function. "
            "Keep its name and signature. "
            "Return the complete Python code, "
            "including any needed imports.\n\n"
        ),
    ),
    # MBPP+ prompts are a docstring with the task and one example assert, which
    # shows the required function name.
    "mbpp": (
        get_mbpp_plus,
        get_mbpp_plus_hash,
        (
            "Solve the following problem in Python. "
            "Use the function name shown in the example. "
            "Return the complete Python code, "
            "including any needed imports.\n\n"
        ),
    ),
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), default="humaneval")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only the first N tasks (for a quick test)"
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=False,
    )

    get_tasks, get_hash, instruction_prefix = DATASETS[args.dataset]
    problems = get_tasks()
    if args.limit:
        problems = dict(list(problems.items())[: args.limit])

    tokenizer, model = load_model(args.model)

    metadata = {
        "model": str(Path(args.model).resolve()),
        "dataset": args.dataset,
        "dataset_hash": get_hash(),
        "task_count": len(problems),
        "packages": {
            package: version(package)
            for package in (
                "torch",
                "transformers",
                "evalplus",
            )
        },
        "gpu": device_name(pick_device()),
        "dtype": "float32",
        "attention": "sdpa",
        "do_sample": False,
        "max_new_tokens": 512,
        "prompt_mode": ("Qwen chat; full function; no assistant prefill"),
        "instruction_prefix": instruction_prefix,
        "extraction": ("evalplus.sanitize.sanitize with required entry point"),
    }

    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
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

            samples.write(json.dumps(sample_record) + "\n")
            raw.write(json.dumps(raw_record) + "\n")

            samples.flush()
            raw.flush()

            print(
                f"{index}/{len(problems)} {task_id} limit={hit_limit}",
                flush=True,
            )

    print("Generation complete. Grade samples.jsonl separately inside Docker.")


if __name__ == "__main__":
    main()
