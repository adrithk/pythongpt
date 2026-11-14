"""Curation step 3 of 3: combine the verified sources into one training set.

In order: format each row the way the trainer expects and drop rows that are
too long, drop repeats, drop anything that looks like a benchmark task, take
each source's quota, and hold out some tested rows for choosing a checkpoint.

Input: data/curated/verified/<source>.jsonl
Output: data/<name>-splits (what the trainer loads), plus the full rows and a
report in data/curated/.
"""

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

from curate_collect import OUTPUT
from distill_common import ROOT, read_jsonl, save_json, write_jsonl

TOKENIZER = ROOT / "models" / "qwen-base"
EOS = "<|im_end|>"
MAX_LENGTH = 1024  # Prompt + completion tokens. Must match the trainer.
# How many rows to take from each source. The order matters: when two rows
# repeat each other, the source listed first keeps its row.
QUOTAS = {
    "deepseek": 30_000,
    "opencode": 120_000,
    "selfoss": 40_000,
    "educational": 50_000,
}
DEV_ROWS = 1000  # Tested rows kept out of training, for choosing a checkpoint.
SIMILAR = 0.6  # Word overlap above which two same-named tasks are repeats.
NAME_CAP = 30  # At most this many rows may define the same function name.
BENCHMARK_SIMILAR = 0.5  # Word overlap with a benchmark task that removes a row.


def words(text):
    return frozenset(re.findall(r"[a-z0-9_]+", text.lower()))


def overlap(left, right):
    """Share of words in common: 0 = nothing shared, 1 = identical sets."""
    return len(left & right) / max(1, len(left | right))


def main_name(code):
    """The name of the first function or class the answer defines."""
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return node.name
    return ""


def add_training_text(rows, tokenizer, reasons):
    """Add the exact prompt and completion strings, and drop rows that are too long.

    Same format as init_prepare.py: the instruction as a user message ending
    with the assistant header, and the code followed by the end marker.
    """
    kept = []
    for start in range(0, len(rows), 2000):
        chunk = rows[start : start + 2000]
        for row in chunk:
            row["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["instruction"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            row["completion"] = row["response"] + EOS
        texts = [row["prompt"] + row["completion"] for row in chunk]
        lengths = [
            len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]
        ]
        for row, length in zip(chunk, lengths):
            if length <= MAX_LENGTH:
                row["tokens"] = length
                kept.append(row)
            else:
                reasons[f"{row['source']}: too long"] += 1
    return kept


def drop_repeats(rows, reasons):
    """Remove exact repeats, near repeats, and overused function names."""
    seen_text, seen_code, by_name, kept = set(), set(), {}, []
    for row in rows:
        text = " ".join(row["instruction"].lower().split())
        code = ast.dump(ast.parse(row["response"]))  # Ignores spacing and comments.
        row["name"], row_words = main_name(row["response"]), words(row["instruction"])
        same_name = by_name.setdefault(row["name"], [])
        if text in seen_text or code in seen_code:
            reason = "exact repeat"
        elif len(same_name) >= NAME_CAP:
            reason = "function name used too often"
        elif any(overlap(row_words, other) >= SIMILAR for other in same_name):
            reason = "near repeat"
        else:
            seen_text.add(text)
            seen_code.add(code)
            same_name.append(row_words)
            kept.append(row)
            continue
        reasons[f"{row['source']}: {reason}"] += 1
    return kept


def load_benchmarks():
    """(function name, words of the task text) for every HumanEval+ and MBPP+ task.

    Used ONLY to remove lookalikes. Benchmark text never enters the training set.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    tasks = [*get_human_eval_plus().values(), *get_mbpp_plus().values()]
    return [(task["entry_point"], words(task["prompt"])) for task in tasks]


def drop_benchmark_lookalikes(rows, benchmarks, reasons):
    """Remove rows that share a benchmark's function name or most of its wording."""
    names = {name for name, _ in benchmarks}
    kept = []
    for row in rows:
        row_words = words(row["instruction"])
        if row["name"] in names:
            reasons[f"{row['source']}: benchmark function name"] += 1
        elif any(
            # Sets of very different sizes cannot overlap enough; skip those early.
            min(len(row_words), len(other))
            >= BENCHMARK_SIMILAR * max(len(row_words), len(other))
            and overlap(row_words, other) >= BENCHMARK_SIMILAR
            for _, other in benchmarks
        ):
            reasons[f"{row['source']}: benchmark wording"] += 1
        else:
            kept.append(row)
    return kept


def build(args):
    splits = ROOT / "data" / f"{args.name}-splits"
    if splits.exists():
        raise RuntimeError(f"{splits} already exists. Choose another --name.")
    rng = random.Random(args.seed)
    rows, reasons, stages = [], Counter(), {}

    def record(stage, current):
        stages[stage] = dict(Counter(row["source"] for row in current))

    for source in QUOTAS:
        path = args.output / "verified" / f"{source}.jsonl"
        if not path.exists():
            print(f"Note: no verified rows for {source}; it is left out.")
            continue
        loaded = read_jsonl(path)
        rng.shuffle(loaded)
        rows.extend(loaded)
    if not rows:
        raise RuntimeError(
            "No verified rows found. Run curate_collect.py and curate_verify.py."
        )
    record("verified", rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    rows = add_training_text(rows, tokenizer, reasons)
    record("fits length", rows)
    rows = drop_repeats(rows, reasons)
    record("after repeats", rows)
    if args.skip_benchmark_audit:
        audit = "NOT RUN"
    else:
        try:
            benchmarks = load_benchmarks()
        except ImportError as exc:
            raise RuntimeError(
                "evalplus is needed for the benchmark check (requirements-eval.txt), "
                "or pass --skip-benchmark-audit."
            ) from exc
        rows = drop_benchmark_lookalikes(rows, benchmarks, reasons)
        audit = f"run against {len(benchmarks)} tasks"
    record("after benchmark check", rows)

    chosen = []
    for source, quota in QUOTAS.items():
        available = [row for row in rows if row["source"] == source]
        chosen.extend(rng.sample(available, min(quota, len(available))))
    rng.shuffle(chosen)
    tested = [row for row in chosen if row["tests"]]
    dev_ids = {
        row["id"] for row in rng.sample(tested, min(DEV_ROWS, len(tested) // 10))
    }
    train = [row for row in chosen if row["id"] not in dev_ids]
    dev = [row for row in chosen if row["id"] in dev_ids]
    record("train", train)
    record("dev", dev)

    from datasets import Dataset, DatasetDict

    def view(part):  # The trainer only needs these two columns.
        return Dataset.from_list(
            [{"prompt": row["prompt"], "completion": row["completion"]} for row in part]
        )

    # The trainer reads the key "test" as its validation set.
    DatasetDict({"train": view(train), "test": view(dev)}).save_to_disk(str(splits))
    write_jsonl(args.output / f"{args.name}-train.jsonl", train)
    write_jsonl(args.output / f"{args.name}-dev.jsonl", dev)
    lengths = sorted(row["tokens"] for row in train)
    report = {
        "rows_per_source_at_each_step": stages,
        "quotas": QUOTAS,
        "removed": dict(sorted(reasons.items())),
        "benchmark_check": audit,
        "train_tokens_total": sum(lengths),
        "train_tokens_median": lengths[len(lengths) // 2] if lengths else 0,
        "most_common_function_names": Counter(row["name"] for row in train).most_common(
            15
        ),
        "seed": args.seed,
    }
    save_json(args.output / f"{args.name}-report.json", report)
    print(json.dumps(report, indent=2))
    print(
        f'\nTrainer data: {splits}\nSet DATA in init_trainsft.py to "data/{args.name}-splits".'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--name", default="s2", help="Names the output: data/<name>-splits"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-benchmark-audit", action="store_true")
    args = parser.parse_args(argv)
    try:
        build(args)
    except (RuntimeError, OSError) as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
