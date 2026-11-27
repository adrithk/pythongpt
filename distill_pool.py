"""Pick the prompts for on-policy distillation (train_distill.py).

    python distill_pool.py

On-policy distillation needs only prompts: the student writes the answers and
the teacher scores every token. So, unlike SFT, a reference solution's quality
does not matter, and the prompts can come from any tested source. A good set:

  - looks like the benchmarks: HumanEval-style stubs (signature + docstring
    with examples) and MBPP-style tasks (a description plus one example
    assert that shows the function name), written exactly as the benchmark
    scripts present them;
  - is new to S3: no KodCode problem S2/S3 trained on, in either of KodCode's
    two styles (Taco_123_C and Taco_123_I are the same problem), and none of
    the GRPO held-out tasks;
  - is spread over topics and difficulty: KodCode's subsets (LeetCode,
    Codeforces, Taco, data structures, algorithms, ...) and its easy/medium/hard
    labels, with medium and hard weighted up; no library-documentation or
    third-party-package tasks;
  - has no repeats and no HumanEval+/MBPP+ lookalikes;
  - keeps its tests, so the GPU step (distill_filter.py) can keep only the
    tasks the TEACHER solves (its token scores are then about correct code)
    and favour the ones S3 gets wrong.

Also adds the 6,416 GRPO pool tasks (already used for GRPO, never for SFT).
Writes data/opd/candidates.jsonl (id, source, style, difficulty,
instruction, tests) and data/opd/candidates-report.json.
Run on the Mac (the benchmark check needs evalplus), then upload.
"""

import ast
import builtins
import glob
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from curate_build import drop_benchmark_lookalikes, drop_repeats, load_benchmarks
from curate_collect import (
    KODCODE,
    clean_code,
    from_kodcode,
    parse_python,
    pytest_to_tests,
)

OUTPUT = Path("data/opd")  # On-policy distillation prompts.
USED = [  # Rows S2/S3 trained or validated on, plus the GRPO held-out tasks.
    "data/curated/s2-200k-train.jsonl",
    "data/curated/s2-200k-dev.jsonl",
    "data/curated-edge4/s3-train.jsonl",
    "data/curated-edge4/s3-dev.jsonl",
    "data/grpo/dev.jsonl",
]
GRPO_POOL = "data/grpo/pool.jsonl"
# KodCode prompts to keep per (style, difficulty) after every filter.
QUOTAS = {
    ("complete", "easy"): 2_000,
    ("complete", "medium"): 3_500,
    ("complete", "hard"): 3_500,
    ("instruct", "easy"): 1_500,
    ("instruct", "medium"): 2_250,
    ("instruct", "hard"): 2_250,
}
INTAKE = 3  # Rows read per kept prompt, since the filters below remove many.
SKIP_SUBSETS = {"Docs", "Package"}  # Library documentation, third-party packages.
MAX_BENCHMARK_SIMILARITY = 0.85  # KodCode's own similarity to benchmark questions.
NAME_CAP = 3
MAX_PROMPT_TOKENS = 512
MAX_ANSWER_TOKENS = 384  # Reference answer; keeps tasks solvable in 512 tokens.
TOKENIZER = "runs/sft-s3/final"
SEED = 0
# Same text benchmark_humaneval.py puts before an MBPP+ prompt.
MBPP_PREFIX = (
    "Solve the following problem in Python. "
    "Use the function name shown in the example. "
    "Return the complete Python code, "
    "including any needed imports.\n\n"
)
SAFE_NAMES = set(dir(builtins))


def used_problems():
    """KodCode problem ids (style suffix removed) that must not be used."""
    problems = set()
    for path in USED:
        with open(path, encoding="utf-8") as file:
            for line in file:
                row_id = json.loads(line)["id"]
                if row_id.startswith("kodcode:"):
                    problems.add(problem_of(row_id[len("kodcode:") :]))
    return problems


def problem_of(question_id):
    return question_id.rsplit("_", 1)[0]  # "Taco_123_C" -> "Taco_123"


def example_assert(test_source, name):
    """A one-line assert from the tests that calls `name` with literal values only.

    MBPP prompts show one such assert; it is how the model learns the name.
    """
    for node in ast.walk(parse_python(test_source)):
        if not isinstance(node, ast.Assert):
            continue
        text = ast.unparse(node)
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        calls_it = f"{name}(" in text
        if calls_it and names <= SAFE_NAMES | {name} and len(text) <= 200:
            return text
    raise ValueError("no self-contained example assert")


def from_kodcode_instruct(row):
    """MBPP-style prompt: the question and one example assert, in a docstring."""
    info = list(row["test_info"])
    if len(info) != 1 or not info[0]["function_name"]:
        raise ValueError("not a single-function task")
    name = info[0]["function_name"]
    tests = pytest_to_tests(row["test"])
    example = example_assert(row["test"], name)
    question = row["question"].strip()
    if "from solution import" in question or "def test" in question:
        raise ValueError("tests left in the question")
    instruction = f'{MBPP_PREFIX}"""\n{question}\n{example}\n"""\n'
    return row["question_id"], instruction, row["solution"], tests


def kodcode_rows(used, rng, reasons):
    columns = [
        "question_id", "style", "subset", "question", "solution", "test",
        "test_info", "gpt_difficulty", "benchmark_similarity",
    ]
    files = sorted(glob.glob(str(KODCODE / "data" / "*.parquet")))
    table = pq.read_table(files, columns=columns).to_pylist()
    rng.shuffle(table)
    taken, problems, rows = Counter(), set(), []
    for row in table:
        key = (row["style"], row["gpt_difficulty"])
        problem = problem_of(row["question_id"])
        if key not in QUOTAS or taken[key] >= INTAKE * QUOTAS[key]:
            continue
        if problem in used:
            reasons["kodcode: S2/S3/GRPO-dev problem"] += 1
            continue
        if problem in problems:
            reasons["kodcode: same problem in the other style"] += 1
            continue
        if row["subset"] in SKIP_SUBSETS:
            reasons["kodcode: docs/package subset"] += 1
            continue
        if row["benchmark_similarity"] > MAX_BENCHMARK_SIMILARITY:
            reasons["kodcode: KodCode benchmark similarity"] += 1
            continue
        try:
            adapter = from_kodcode if row["style"] == "complete" else from_kodcode_instruct
            question_id, instruction, solution, tests = adapter(row)
            response = clean_code(solution)
        except (ValueError, SyntaxError) as error:
            reasons[f"kodcode {row['style']}: {error}"] += 1
            continue
        taken[key] += 1
        problems.add(problem)
        rows.append(
            {
                "id": f"kodcode:{question_id}",
                "source": "kodcode",
                "style": row["style"],
                "difficulty": row["gpt_difficulty"],
                "subset": row["subset"],
                "instruction": instruction.strip() + "\n",
                "response": response,
                "tests": tests[:20],
            }
        )
    print("KodCode taken:", dict(taken))
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "candidates.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; move it away to rebuild.")
    rng = random.Random(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    reasons = Counter()
    rows = kodcode_rows(used_problems(), rng, reasons)
    reasons["kodcode taken"] = len(rows)

    kept = []
    for row in rows:
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
    per_key, trimmed = Counter(), []
    for row in kept:  # Rows are in random order, so this keeps a random subset.
        key = (row["style"], row["difficulty"])
        if per_key[key] < QUOTAS[key]:
            per_key[key] += 1
            trimmed.append(row)
    reasons["over quota after filters"] = len(kept) - len(trimmed)
    kept = trimmed

    with open(GRPO_POOL, encoding="utf-8") as file:
        grpo = [json.loads(line) for line in file]
    for task in grpo:  # Already filtered the same way by grpo_pool.py.
        task.update(style="grpo-pool", difficulty="", subset="")
    tasks = [
        {key: row[key] for key in
         ("id", "source", "style", "difficulty", "subset", "instruction", "tests")}
        for row in kept + grpo
    ]
    rng.shuffle(tasks)
    with out.open("w", encoding="utf-8") as file:
        for task in tasks:
            file.write(json.dumps(task, ensure_ascii=False) + "\n")
    report = {
        "removed": dict(reasons.most_common()),
        "candidates": len(tasks),
        "by_style_difficulty": dict(
            Counter(f"{t['style']}/{t['difficulty']}" for t in tasks).most_common()
        ),
        "by_subset": dict(Counter(t["subset"] for t in tasks).most_common()),
        "settings": {
            "quotas": {"/".join(k): v for k, v in QUOTAS.items()},
            "skip_subsets": sorted(SKIP_SUBSETS),
            "max_benchmark_similarity": MAX_BENCHMARK_SIMILARITY,
            "name_cap": NAME_CAP,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "seed": SEED,
        },
    }
    (OUTPUT / "candidates-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
