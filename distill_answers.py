"""Stage 2 of 3: solve the questions.

For each question DeepSeek returns a solution AND tests for it, in one call.
Nothing is run here: distill_verify.py runs the tests afterwards. If a
solution fails there, running this stage again asks for one fresh attempt.

Input: questions.jsonl. Output: one file per attempt in answers/.
"""

import argparse
import ast
import math
from pathlib import Path

from distill_common import (
    CONTROL_TOKENS,
    DEFAULT_OUTPUT,
    MAX_OUTPUT_TOKENS,
    OUTPUT_PRICE,
    Budget,
    ask_json,
    finish,
    load_json,
    locked,
    make_client,
    parse_python,
    read_jsonl,
    run_parallel,
    save_json,
)

MAX_ATTEMPTS = 2  # The first answer plus one retry.
MIN_TESTS, MAX_TESTS = 5, 12
PROMPT = """Solve this Python problem, then write tests for your solution.

--- problem ---
{instruction}
--- end of problem ---
{extra}
Return JSON: {{"code": "...", "tests": ["assert ...", "assert ..."]}}

code: the complete solution as plain Python without Markdown fences. It must
define `{entry}`, keep any signature the problem gives, and include the imports
it needs. Write it the way a careful programmer would: clear, direct, standard
library only. No printing, no example calls, no comments that restate the code,
and do not repeat a docstring the problem already gave.

tests: 6 to 10 separate one-line `assert` statements that call `{entry}`. Cover
ordinary cases, the edge cases the problem mentions (empty input, ties,
boundaries) and one larger case. Work each expected value out carefully, since
a wrong test throws away a correct solution. Compare floats with a tolerance.
"""
FIX_EXTRA = "\nAt least one of your tests must fail for the buggy code shown.\n"
RETRY_EXTRA = """
An earlier attempt at this problem was rejected: {failure}
The mistake may have been in the solution or in a test. Start again carefully.
"""


def make_prompt(question, failure=None):
    extra = FIX_EXTRA if question["form"] == "fix" else ""
    if failure:
        extra += RETRY_EXTRA.format(failure=failure)
    return PROMPT.format(
        instruction=question["instruction"],
        entry=question["entry_point"],
        extra=extra,
    )


def check_answer(entry, code, tests):
    """Read-only checks. Returns the cleaned (code, tests) or raises ValueError."""
    code = code.strip()
    if code.startswith("```"):  # Remove Markdown fences if the model added them.
        code = "\n".join(code.splitlines()[1:]).rsplit("```", 1)[0].strip()
    if not code or any(token in code for token in CONTROL_TOKENS):
        raise ValueError("empty code or unexpected chat control token")
    tree = parse_python(code)
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == entry for node in tree.body
    ):
        raise ValueError("solution does not define the requested function")
    cleaned = []
    for test in tests:
        if not isinstance(test, str):
            raise ValueError("a test is not a string")  # noqa: TRY004
        statements = parse_python(test.strip()).body
        if len(statements) != 1 or not isinstance(statements[0], ast.Assert):
            raise ValueError("a test is not a single assert statement")
        names = {
            node.id for node in ast.walk(statements[0]) if isinstance(node, ast.Name)
        }
        if entry not in names:
            raise ValueError("a test does not call the function")
        if test.strip() not in cleaned:
            cleaned.append(test.strip())
    if len(cleaned) < MIN_TESTS:
        raise ValueError(f"fewer than {MIN_TESTS} distinct tests")
    return code, cleaned[:MAX_TESTS]


def next_attempt(output, question_id):
    """Decide what this question needs.

    Returns (attempt number, why the last attempt failed), or None when the
    question is finished, out of attempts, or waiting for distill_verify.py.
    """
    failure = None
    for attempt in range(MAX_ATTEMPTS):
        name = f"{question_id}_{attempt}.json"
        if not (output / "answers" / name).exists():
            return attempt, failure
        answer = load_json(output / "answers" / name)
        if answer["status"] == "rejected":
            failure = answer["reason"]
            continue
        if not (output / "verified" / name).exists():
            return None  # Not tested yet.
        result = load_json(output / "verified" / name)
        if result["passed"]:
            return None
        failure = result["reason"]
    return None


def collect(client, args):
    output = args.output
    if not (output / "questions.jsonl").exists():
        raise RuntimeError("No questions.jsonl here. Run distill_questions.py first.")
    budget = Budget(output, args.budget_usd)
    todo = [
        question
        for question in read_jsonl(output / "questions.jsonl")
        if next_attempt(output, question["id"]) is not None
    ]
    print(f"{len(todo)} questions need an answer.")

    def work(question):
        made = []
        while (step := next_attempt(output, question["id"])) is not None:
            attempt, failure = step
            name = f"{question['id']}_{attempt}"
            reply, receipt = ask_json(
                client,
                budget,
                output,
                f"answer_{name}",
                make_prompt(question, failure),
                {"code": str, "tests": list},
                args.effort,
            )
            record = {"id": question["id"], "attempt": attempt, "provenance": receipt}
            try:
                code, tests = check_answer(
                    question["entry_point"], reply["code"], reply["tests"]
                )
                record.update(status="unverified", code=code, tests=tests)
            except ValueError as exc:
                record.update(status="rejected", reason=str(exc))
            save_json(output / "answers" / f"{name}.json", record)
            made.append(record["status"])
        return f"{question['id']}: {', '.join(made)}; USD {budget.spent:.4f}"

    failed = run_parallel(work, todo, args.workers)
    print("Next: python distill_verify.py  (runs the tests; needs Docker)")
    if failed:
        raise RuntimeError(f"{failed} questions failed; repeat the command to retry.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget-usd", type=float, help="Spending ceiling, all stages")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan only")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if args.dry_run:
        questions = read_jsonl(args.output / "questions.jsonl")
        todo = [q for q in questions if next_attempt(args.output, q["id"]) is not None]
        worst = len(todo) * MAX_OUTPUT_TOKENS * OUTPUT_PRICE / 1e6
        print(
            f"No API calls made. {len(todo)} of {len(questions)} questions need an answer."
        )
        print(
            f"Worst-case cost if every call used its whole allowance: USD {worst:.2f}"
        )
        if todo:
            print("\n--- prompt for the first one ---\n")
            print(make_prompt(todo[0], next_attempt(args.output, todo[0]["id"])[1]))
        return 0
    if args.budget_usd is None or not 0 < args.budget_usd < math.inf:
        parser.error("--budget-usd must be a positive number.")

    def stage():
        with locked(args.output):
            collect(make_client(), args)

    return finish(stage)


if __name__ == "__main__":
    raise SystemExit(main())
