"""Stage 1 of 3: write the questions.

The PROGRAM decides what each question is about (skill, form, difficulty) and
DeepSeek only writes it. That is where the variety comes from: asking a model
to "be diverse" gives repeats, so the mix is fixed here in ordinary Python.

Output: questions.jsonl in the output folder. Nothing is solved or run here.
"""

import argparse
import ast
import json
import math
import random
import re
import threading
from pathlib import Path

from distill_common import (
    CONTROL_TOKENS,
    DEFAULT_OUTPUT,
    MAX_OUTPUT_TOKENS,
    OUTPUT_PRICE,
    ROOT,
    Budget,
    ask_json,
    finish,
    load_json,
    locked,
    make_client,
    parse_python,
    run_parallel,
    save_json,
    write_jsonl,
)

SKILLS_FILE = ROOT / "distill_skills.txt"
BATCH = 10  # Questions written per API call.
EFFORT = "low"  # Writing questions needs little reasoning; solving needs more.
SIMILAR = 0.6  # Share of words in common above which two questions are repeats.

# The mix, as weights. Change the numbers to change the dataset.
FORMS = {"describe": 60, "complete": 25, "fix": 15}
LEVELS = {"easy": 35, "medium": 45, "hard": 20}

FORM_HELP = {
    "describe": "text = a plain-language request that names the function and its "
    'parameters. code = "".',
    "complete": 'text = "". code = ONLY a Python signature with type hints and a '
    'docstring that states the behaviour and shows 2-3 ">>>" examples. No body.',
    "fix": "text = what the function is supposed to do, without saying what is "
    "wrong. code = a complete function that looks reasonable but has ONE "
    "realistic bug that gives wrong results for some inputs.",
}
LEVEL_HELP = {
    "easy": "one idea; a good solution is about 3-8 lines",
    "medium": "two ideas combined, or one tricky case; about 8-20 lines",
    "hard": "a real algorithm or careful bookkeeping; about 20-40 lines",
}
# The program wraps "complete" and "fix" questions in one of these sentences,
# so the student sees several phrasings of the same kind of request.
COMPLETE_INTROS = [
    (
        "Implement the following Python function. Keep its name and signature. "
        "Return the complete Python code, including any needed imports."
    ),
    "Complete this Python function. Do not change the signature.",
    "Write the body of the function below and return the whole function.",
    "Finish this function so that it behaves as its docstring describes.",
]
FIX_INTROS = [
    "The function below has a bug. Return the corrected, complete function.",
    (
        "This implementation gives wrong results for some inputs. Fix it and "
        "return the full function."
    ),
    "Find and fix the bug in this code. Keep the function name and signature.",
]
RULES = """You write practice problems for training a small Python coding model.
Every problem asks for ONE self-contained Python function that is:
- pure and deterministic: no files, network, input(), printing, randomness,
  current time or global state;
- standard library only, taking and returning plain Python values (numbers,
  strings, booleans, None, lists, tuples, dicts, sets), so that a test can be
  written as `assert f(arguments) == expected`;
- fully specified: say exactly what is returned, including for empty input and
  for ties whenever those can happen;
- about the skill, not about input checking: never ask for type checks, and
  never ask the function to raise an exception;
- original: do not reproduce problems from HumanEval, MBPP, LeetCode or any
  other benchmark or website.
"""


def load_skills(path=SKILLS_FILE):
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    skills = [line for line in lines if line and not line.startswith("#")]
    if not skills:
        raise ValueError(f"No skills found in {path}")
    return skills


def plan_batch(number, skills, seed):
    """Choose one skill for batch `number`, plus a form and level per question.

    The skill order is one fixed shuffle of the whole list, so every skill is
    used once before any repeats. The same seed always gives the same plan.
    """
    order = random.Random(seed).sample(skills, len(skills))
    rng = random.Random(f"{seed}:{number}")
    slots = [
        {
            "form": rng.choices(list(FORMS), weights=list(FORMS.values()))[0],
            "level": rng.choices(list(LEVELS), weights=list(LEVELS.values()))[0],
        }
        for _ in range(BATCH)
    ]
    return order[number % len(order)], slots


def make_prompt(skill, slots, used_names):
    numbered = "\n".join(
        f"{index}. form={slot['form']}, difficulty={slot['level']}"
        for index, slot in enumerate(slots, start=1)
    )
    forms = "\n".join(f"- {name}: {text}" for name, text in FORM_HELP.items())
    levels = "\n".join(f"- {name}: {text}" for name, text in LEVEL_HELP.items())
    return f"""{RULES}
Skill for this batch: {skill}

Write {len(slots)} problems that all use this skill but ask for clearly
different computations. Problem N must have the form and difficulty of slot N:
{numbered}

Forms:
{forms}

Difficulty:
{levels}

Vary the wording, the kinds of input and the function names. Function names
already used for this skill, which you must not reuse or lightly rename:
{json.dumps(sorted(used_names))}

Return JSON in slot order:
{{"questions": [{{"entry_point": "function_name", "text": "...", "code": "..."}}]}}
"""


def check_question(form, entry, text, code):
    """Cheap checks that read the question but never run any code."""
    if not entry.isidentifier():
        raise ValueError("invalid function name")
    if any(token in text + code for token in CONTROL_TOKENS):
        raise ValueError("unexpected chat control token")
    if form == "describe":
        if not re.search(rf"\b{entry}\b", text):
            raise ValueError("request does not name the function")
        return
    tree = parse_python(code)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entry
    ]
    if not functions:
        raise ValueError("code does not define the function")
    if form == "fix" and not text:
        raise ValueError("missing description of the intended behaviour")
    if form == "complete":
        if not ast.get_docstring(functions[0]):
            raise ValueError("stub has no docstring")
        if len(functions[0].body) > 2:  # Docstring plus at most `pass` or `...`.
            raise ValueError("stub already contains a solution")


def build_instruction(form, text, code, rng):
    """Turn the model's pieces into the exact text the student will be shown."""
    if form == "describe":
        return text
    if form == "complete":
        return f"{rng.choice(COMPLETE_INTROS)}\n\n{code}"
    return f"{text}\n\n{rng.choice(FIX_INTROS)}\n\n```python\n{code}\n```"


class Seen:
    """Remembers kept questions so that repeats can be rejected.

    A new question is compared with earlier ones that share its skill or its
    function name. Two questions are repeats when most of their words match.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.groups = {}  # "skill:..." or "name:..." -> list of word sets
        self.names = {}  # skill -> function names already used

    def add(self, skill, entry, text, code):
        words = frozenset(re.findall(r"[a-z0-9_]+", (text + " " + code).lower()))
        keys = (f"skill:{skill}", f"name:{entry}")
        with self.lock:
            for key in keys:
                for other in self.groups.get(key, []):
                    if len(words & other) >= SIMILAR * len(words | other):
                        raise ValueError("too similar to an earlier question")
            for key in keys:
                self.groups.setdefault(key, []).append(words)
            self.names.setdefault(skill, set()).add(entry)

    def used_names(self, skill):
        with self.lock:
            return set(self.names.get(skill, ()))


def collect(client, args):
    output = args.output
    skills = load_skills()
    budget = Budget(output, args.budget_usd)
    seen = Seen()
    for path in sorted((output / "questions").glob("*.json")):
        for row in load_json(path)["records"]:
            if row["status"] == "ok":
                seen.add(row["skill"], row["entry_point"], row["text"], row["code"])

    def work(number):
        skill, slots = plan_batch(number, skills, args.seed)
        reply, receipt = ask_json(
            client,
            budget,
            output,
            f"questions_{number:05d}",
            make_prompt(skill, slots, seen.used_names(skill)),
            {"questions": list},
            EFFORT,
        )
        rng = random.Random(f"{args.seed}:intro:{number}")
        records = []
        for index, (slot, row) in enumerate(zip(slots, reply["questions"])):
            record = {"id": f"q{number:05d}_{index}", "skill": skill, **slot}
            try:
                if not isinstance(row, dict):
                    raise ValueError("question is not a JSON object")  # noqa: TRY004
                entry, text, code = (
                    str(row.get(key, "")).strip()
                    for key in ("entry_point", "text", "code")
                )
                if slot["form"] == "describe":
                    code = ""
                check_question(slot["form"], entry, text, code)
                seen.add(skill, entry, text, code)
                record.update(
                    status="ok",
                    entry_point=entry,
                    text=text,
                    code=code,
                    instruction=build_instruction(slot["form"], text, code, rng),
                )
            except ValueError as exc:
                record.update(status="rejected", reason=str(exc))
            records.append(record)
        save_json(
            output / "questions" / f"{number:05d}.json",
            {"records": records, "provenance": receipt},
        )
        kept = sum(record["status"] == "ok" for record in records)
        return f"batch {number} ({skill}): kept {kept}/{BATCH}; USD {budget.spent:.4f}"

    total = math.ceil(args.questions / BATCH)
    failed = 0
    try:
        # One pass over the skill list at a time. Batches inside a pass have
        # different skills, so they can run in parallel; the next pass then
        # sees which function names each skill has already used.
        for start in range(0, total, len(skills)):
            todo = [
                number
                for number in range(start, min(start + len(skills), total))
                if not (output / "questions" / f"{number:05d}.json").exists()
            ]
            failed += run_parallel(work, todo, args.workers)
    finally:
        rows = [
            row
            for path in sorted((output / "questions").glob("*.json"))
            for row in load_json(path)["records"]
            if row["status"] == "ok"
        ]
        write_jsonl(output / "questions.jsonl", rows)
        print(f"{len(rows)} questions in {output / 'questions.jsonl'}")
    if failed:
        raise RuntimeError(f"{failed} batches failed; repeat the command to retry.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=int, default=200, help="Total to ask for")
    parser.add_argument("--budget-usd", type=float, help="Spending ceiling, all stages")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan only")
    args = parser.parse_args(argv)
    if args.questions < 1 or args.workers < 1:
        parser.error("--questions and --workers must be positive.")
    if args.dry_run:
        skills = load_skills()
        calls = math.ceil(args.questions / BATCH)
        worst = calls * MAX_OUTPUT_TOKENS * OUTPUT_PRICE / 1e6
        print(f"No API calls made. {len(skills)} skills; {calls} calls planned.")
        print(
            f"Worst-case cost if every call used its whole allowance: USD {worst:.2f}"
        )
        print("\n--- prompt for batch 0 ---\n")
        print(make_prompt(*plan_batch(0, skills, args.seed), set()))
        return 0
    if args.budget_usd is None or not 0 < args.budget_usd < math.inf:
        parser.error("--budget-usd must be a positive number.")

    def stage():
        with locked(args.output):
            collect(make_client(), args)

    return finish(stage)


if __name__ == "__main__":
    raise SystemExit(main())
