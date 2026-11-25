"""Generate answers for LiveCodeBench (v6 file: contests January-April 2025).

    python benchmark_lcb.py --model runs/sft-s3/final --out runs/benchmarks/s3-lcb

LiveCodeBench problems were never used to build or filter our training data,
so this is the untouched test. Two kinds of problem:
- LeetCode ("functional"): complete a method of `class Solution` (our training shape).
- AtCoder ("stdin"): a whole program that reads input and prints the answer
  (a shape our training data does not contain).
The prompt follows LiveCodeBench's own wording. Grade with grade_lcb.py.
"""

import argparse
import json
import re
from pathlib import Path

from infer import device_name, generate, load_model, pick_device

DATA = Path(__file__).resolve().parent / "data" / "lcb-raw" / "test6.jsonl"
MAX_NEW_TOKENS = 1024  # Whole programs are longer than one function.

FUNCTIONAL = (
    "### Format: You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters.\n```python\n{starter}\n```"
)
STDIN = (
    "### Format: Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```"
)


def instruction(row):
    starter = row["starter_code"].strip()
    form = FUNCTIONAL.format(starter=starter) if starter else STDIN
    return (
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests.\n\n"
        f"### Question:\n{row['question_content']}\n\n{form}\n\n"
        "### Answer: (use the provided format with backticks)\n"
    )


def extract(answer):
    """The last fenced Python block, or the whole answer if there is none."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    return blocks[-1] if blocks else answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    rows = [json.loads(line) for line in DATA.open()]
    if args.limit:
        rows = rows[: args.limit]
    tokenizer, model = load_model(args.model)
    revision = (DATA.parent / "REVISION").read_text().strip()
    metadata = {
        "model": str(Path(args.model).resolve()),
        "dataset": "livecodebench/code_generation_lite test6.jsonl",
        "revision": revision,
        "problems": len(rows),
        "gpu": device_name(pick_device()),
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with (args.out / "answers.jsonl").open("x") as file:
        for index, row in enumerate(rows, start=1):
            answer, hit_limit = generate(
                tokenizer, model, instruction(row), max_new_tokens=MAX_NEW_TOKENS
            )
            record = {
                "question_id": row["question_id"],
                "platform": row["platform"],
                "difficulty": row["difficulty"],
                "answer": answer,
                "code": extract(answer),
                "hit_limit": hit_limit,
            }
            file.write(json.dumps(record) + "\n")
            file.flush()
            print(f"{index}/{len(rows)} {row['platform']} {row['difficulty']}", flush=True)
    print("Done. Grade with grade_lcb.py.")


if __name__ == "__main__":
    main()
