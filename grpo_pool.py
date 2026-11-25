"""Pick the tasks for GRPO training: verified rows that S2 and S3 never saw.

    python grpo_pool.py

Takes a random sample of the Docker-verified KodCode and OpenCodeInstruct rows
in data/curated-edge4/verified that were not in S2's or S3's training or
validation sets, then drops rows that look like a HumanEval+/MBPP+ task, have
no tests, repeat a function name too often, or are too long. Writes:

    data/grpo/pool.jsonl   tasks to train on (id, source, instruction, tests)
    data/grpo/dev.jsonl    DEV_TASKS held-out tasks for checking progress
    data/grpo/report.json  how many rows each step removed

Run it on the Mac (the benchmark check needs evalplus), then upload data/grpo.
"""

import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from curate_build import drop_benchmark_lookalikes, drop_repeats, load_benchmarks

VERIFIED = Path("data/curated-edge4/verified")
USED = [  # Every row S2 or S3 trained or validated on.
    "data/curated/s2-200k-train.jsonl",
    "data/curated/s2-200k-dev.jsonl",
    "data/curated-edge4/s3-train.jsonl",
    "data/curated-edge4/s3-dev.jsonl",
]
SOURCES = ["kodcode", "opencode"]
SAMPLE = 15_000  # Random unused rows drawn per source, before the filters.
NAME_CAP = 3  # At most this many tasks may define the same function name.
MAX_PROMPT_TOKENS = 512
MAX_ANSWER_TOKENS = 384  # Reference answer; keeps tasks solvable in 512 tokens.
DEV_TASKS = 200
OUTPUT = Path("data/grpo")
TOKENIZER = "runs/sft-s3/final"
SEED = 0


def used_ids():
    ids = set()
    for path in USED:
        with open(path, encoding="utf-8") as file:
            for line in file:
                ids.add(json.loads(line)["id"])
    return ids


def random_unused_rows(source, used, rng):
    """A uniform random SAMPLE of the source's unused rows ("reservoir sampling").

    The OpenCodeInstruct file is 2.4 GB, so rows are kept as text until picked.
    """
    picked, seen = [], 0
    with open(VERIFIED / f"{source}.jsonl", encoding="utf-8") as file:
        for line in file:
            row_id = line[len('{"id": "') : line.index('"', len('{"id": "'))]
            if row_id in used:
                continue
            seen += 1
            if len(picked) < SAMPLE:
                picked.append(line)
            else:  # Replace a kept row with chance SAMPLE / seen.
                slot = rng.randrange(seen)
                if slot < SAMPLE:
                    picked[slot] = line
    print(f"{source}: {seen} unused rows, sampled {len(picked)}")
    return [json.loads(line) for line in picked]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)  # Never overwrite a pool in use.
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    used = used_ids()
    print("Rows used by S2/S3:", len(used))

    reasons = Counter()
    rows = []
    for source in SOURCES:
        rows += random_unused_rows(source, used, rng)
    reasons["sampled"] = len(rows)

    with_tests = [row for row in rows if row["tests"]]
    reasons["no tests"] = len(rows) - len(with_tests)

    kept = []
    for row in with_tests:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tokenizer(prompt)["input_ids"]) > MAX_PROMPT_TOKENS:
            reasons["prompt too long"] += 1
        elif len(tokenizer(row["response"])["input_ids"]) > MAX_ANSWER_TOKENS:
            reasons["reference answer too long"] += 1
        else:
            kept.append(row)

    kept = drop_repeats(kept, reasons, name_cap=NAME_CAP)
    kept = drop_benchmark_lookalikes(kept, load_benchmarks(), reasons)
    rng.shuffle(kept)

    tasks = [
        {key: row[key] for key in ("id", "source", "instruction", "tests")}
        for row in kept
    ]
    parts = {"dev": tasks[:DEV_TASKS], "pool": tasks[DEV_TASKS:]}
    for name, part in parts.items():
        with (OUTPUT / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for task in part:
                file.write(json.dumps(task, ensure_ascii=False) + "\n")
    report = {
        "removed": dict(reasons.most_common()),
        "pool": dict(Counter(task["source"] for task in parts["pool"])),
        "dev": dict(Counter(task["source"] for task in parts["dev"])),
        "settings": {
            "sample_per_source": SAMPLE,
            "name_cap": NAME_CAP,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "seed": SEED,
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
