"""Curation step 1 of 3: pull each source and convert it to one row shape.

Every source stores its questions, answers and tests differently. Each small
`from_...` function below reads one source's row and hands back the same four
things: an id, the instruction, the code, and the tests. Shared checks then
clean the code and reject rows that are out of scope. Nothing is executed.

Output: data/curated/candidates/<source>.jsonl
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from distill_common import CONTROL_TOKENS, ROOT, parse_python, read_jsonl, save_json

OUTPUT = ROOT / "data" / "curated"
DISTILLED = ROOT / "data" / "distill" / "accepted.jsonl"
MAX_CHARS = 6000  # Rough early limit; the exact token limit is applied in step 3.
MAX_TESTS = 20
MIN_JUDGE = 5  # OpenCodeInstruct: every judge score must be at least this (max 5).
# Standard-library modules that reach outside the function: files, network,
# processes, the clock, randomness. Code that imports them is out of scope.
BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "tempfile",
    "glob",
    "threading",
    "multiprocessing",
    "asyncio",
    "concurrent",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "ctypes",
    "signal",
    "pickle",
    "sqlite3",
    "tkinter",
    "turtle",
    "logging",
    "argparse",
    "random",
    "time",
    "secrets",
    "uuid",
    "io",
}
BLOCKED_CALLS = {"input", "open", "exec", "eval", "exit", "quit", "breakpoint"}
# A Markdown code block: ```python ... ``` (or with no language written).
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def clean_code(code):
    """Keep only imports, functions, classes and constants. Reject out-of-scope code.

    Many answers end with example calls or prints. Those are dropped, so the
    training answer is just the definitions.
    """
    tree = parse_python(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." if node.level else node.module]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                raise ValueError(f"uses {node.func.id}()")
            continue
        else:
            continue
        for module in modules:
            root = module.split(".")[0]
            if root not in sys.stdlib_module_names or root in BLOCKED_MODULES:
                raise ValueError("import outside the allowed standard library")
    lines, pieces, has_definition = code.splitlines(), [], False
    for node in tree.body:
        small = isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            has_definition = True
        elif not small:
            continue  # Example calls, prints, `if __name__ == "__main__":` blocks.
        first = min(
            [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
        )
        text = "\n".join(lines[first - 1 : node.end_lineno])
        # Imports and constants sit on adjacent lines; definitions get space.
        joiner = "\n" if small and pieces and pieces[-1][0] else "\n\n\n"
        pieces.append((small, joiner, text))
    if not has_definition:
        raise ValueError("no function or class")
    return "".join(joiner + text for _, joiner, text in pieces).strip()


def clean_tests(tests):
    """Each test must be valid Python. They are run later, in step 2."""
    cleaned = [test.strip() for test in tests if isinstance(test, str) and test.strip()]
    for test in cleaned:
        parse_python(test)
    return cleaned[:MAX_TESTS]


def make_row(source, row_id, instruction, code, tests):
    """The one row shape every later step uses."""
    instruction = instruction.strip()
    if not instruction or len(instruction) + len(code) > MAX_CHARS:
        raise ValueError("empty or too long")
    if any(token in instruction + code for token in CONTROL_TOKENS):
        raise ValueError("chat control token in text")
    return {
        "id": f"{source}:{row_id}",
        "source": source,
        "instruction": instruction,
        "response": clean_code(code),
        "tests": clean_tests(tests),
    }


def first_code_block(text):
    blocks = FENCE.findall(text)
    if not blocks:
        raise ValueError("no code block")
    return blocks[0]


def parse_structure(text):
    """Some fields store a list or dictionary as text. Turn it back into one."""
    try:
        return json.loads(text)
    except ValueError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("unreadable stored field") from exc


# ---- one small adapter per source: row -> (id, instruction, code, tests)


def from_educational(row):
    tests = [test for test in row["testcase"] if test.strip()]
    if not all(test.lstrip().startswith("assert") for test in tests):
        tests = ["\n".join(row["testcase"])]  # Setup lines: run them as one block.
    return row["seq_id"], row["instruction"], row["code"], tests


def from_opencode(row):
    if float(row["average_test_score"]) != 1.0:
        raise ValueError("a recorded test failed upstream")
    judgement = parse_structure(row["llm_judgement"])
    scores = [item.get("score") for item in judgement.values()]
    if not scores or any(
        type(score) is not int or score < MIN_JUDGE for score in scores
    ):
        raise ValueError("judge score below the minimum")
    tests = parse_structure(row["unit_tests"])
    return row["id"], row["input"], first_code_block(row["output"]), tests


def from_selfoss(row):
    blocks = FENCE.findall(row["response"])
    if not blocks:
        raise ValueError("no code block")
    tests = [
        line.strip()
        for block in blocks[1:]
        for line in block.splitlines()
        if line.startswith("assert ")
    ]
    return row["id"], row["instruction"], blocks[0], tests


def from_deepseek(row):
    return row["id"], row["instruction"], row["response"], row["tests"]


# name -> (Hugging Face dataset, configuration, adapter)
SOURCES = {
    "opencode": ("nvidia/OpenCodeInstruct", None, from_opencode),
    "educational": (
        "OpenCoder-LLM/opc-sft-stage2",
        "educational_instruct",
        from_educational,
    ),
    "selfoss": ("bigcode/self-oss-instruct-sc2-exec-filter-50k", None, from_selfoss),
    "deepseek": (None, None, from_deepseek),
}


def stream(source, seed, shuffle):
    """Yield the source's raw rows one at a time, without downloading it all first."""
    dataset, config, _ = SOURCES[source]
    if dataset is None:
        if not DISTILLED.exists():
            raise RuntimeError(f"{DISTILLED} not found. Run the distill stages first.")
        yield from read_jsonl(DISTILLED)
        return
    from datasets import load_dataset

    rows = load_dataset(dataset, config, split="train", streaming=True)
    if shuffle:  # Mixes the order so a partial pull is not just the first rows.
        rows = rows.shuffle(seed=seed, buffer_size=10_000)
    yield from rows


def convert(source, raw_rows, take):
    """Run the adapter and checks on each row. Returns (kept rows, reason counts)."""
    adapter, kept, reasons = SOURCES[source][2], [], Counter()
    for raw in raw_rows:
        try:
            kept.append(make_row(source, *adapter(raw)))
            reasons["kept"] += 1
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            reasons[
                str(exc) if isinstance(exc, ValueError) else "unexpected row shape"
            ] += 1
        if take and len(kept) >= take:
            break
    return kept, reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", choices=list(SOURCES))
    parser.add_argument(
        "--take", type=int, default=0, help="Stop after this many kept rows (0 = all)"
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        help="Only try the first N rows and print what happens",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.peek:
            raw = (
                row
                for _, row in zip(
                    range(args.peek), stream(args.source, args.seed, False)
                )
            )
            kept, reasons = convert(args.source, raw, 0)
            print(json.dumps(dict(reasons.most_common()), indent=2))
            if kept:
                print("\n--- one kept row ---\n" + json.dumps(kept[0], indent=2))
            return 0
        kept, reasons = convert(
            args.source, stream(args.source, args.seed, True), args.take
        )
    except (RuntimeError, OSError, ImportError) as exc:
        print(f"Stopped: {exc}")
        return 1
    path = args.output / "candidates" / f"{args.source}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("w", encoding="utf-8") as file:
        for row in kept:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(".tmp").replace(path)
    save_json(path.with_suffix(".report.json"), dict(reasons.most_common()))
    print(json.dumps(dict(reasons.most_common()), indent=2))
    print(f"{len(kept)} candidate rows in {path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    # Everything is saved by now. Leave at once: after a partial streaming read
    # the download library's background threads can hang a normal shutdown.
    os._exit(exit_code)
