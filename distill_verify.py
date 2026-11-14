"""Stage 3 of 3: run the tests and keep only what passes.

Each solution runs against its tests inside a throwaway Docker container with
no network, no writable disk and a time limit. No API calls, so no cost.

Input: answers/. Output: accepted.jsonl (the training candidates) and report.json.
"""

import argparse
import json
import secrets
import shlex
import subprocess
from collections import Counter
from pathlib import Path

from distill_answers import MAX_ATTEMPTS
from distill_common import (
    DEFAULT_OUTPUT,
    Budget,
    finish,
    load_json,
    locked,
    read_jsonl,
    run_parallel,
    save_json,
    write_jsonl,
)

IMAGE = "python:3.12-slim"
TIME_LIMIT = 10  # Seconds for one solution and all of its tests.

# This small program runs INSIDE the container. It receives the code and tests
# as JSON on standard input and prints one line: which tests raised an error.
RUNNER = r"""
import contextlib, io, json, sys
job = json.load(sys.stdin)
errors = []
for test in job["tests"]:
    space = {"__name__": "candidate"}  # A fresh namespace for every test.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(job["code"], space)
            exec(test, space)
        errors.append(None)
    except BaseException as exc:
        errors.append(type(exc).__name__)
sys.stdout.write(job["marker"] + json.dumps(errors) + "\n")
"""


def docker_command(docker, image):
    """The container is deliberately given almost nothing."""
    return [
        *docker,
        "run",
        "--rm",
        "--interactive",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=256m",
        "--pids-limit=64",
        "--cpus=1",
        image,
        "timeout",
        "--signal=KILL",
        str(TIME_LIMIT),
        "python",
        "-I",
        "-c",
        RUNNER,
    ]


def run_tests(command, code, tests):
    """Return one entry per test (None = passed, else the error name).

    Returns None instead if the whole run crashed or hit the time limit.
    """
    marker = secrets.token_hex(8)  # So printed output cannot be mistaken for results.
    job = json.dumps({"code": code, "tests": tests, "marker": marker})
    try:
        done = subprocess.run(
            command,
            input=job,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT + 60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    return None


def judge(command, question, answer):
    """Decide whether one answer is kept. The reason is shown to a retry."""
    errors = run_tests(command, answer["code"], answer["tests"])
    if errors is None:
        return {"passed": False, "reason": "the solution crashed or timed out"}
    for test, error in zip(answer["tests"], errors):
        if error:
            return {"passed": False, "reason": f"{error} in test: {test}"}
    if question["form"] == "fix":
        # The given buggy code must fail at least one test; otherwise the
        # tests would accept the bug and prove nothing.
        buggy = run_tests(command, question["code"], answer["tests"])
        if buggy is not None and not any(buggy):
            return {"passed": False, "reason": "no test fails for the buggy code"}
    return {"passed": True, "reason": ""}


def export(output):
    """Write accepted.jsonl and a small report of what happened to each question."""
    accepted, counts = [], Counter()
    questions = read_jsonl(output / "questions.jsonl")
    for question in questions:
        group = f"{question['form']}/{question['level']}"
        outcome = "no answer yet"
        for path in sorted((output / "answers").glob(f"{question['id']}_*.json")):
            answer = load_json(path)
            result = output / "verified" / path.name
            if answer["status"] == "rejected":
                outcome = "failed"
            elif not result.exists():
                outcome = "not tested yet"
            elif load_json(result)["passed"]:
                outcome = "accepted"
                accepted.append(
                    {
                        "id": question["id"],
                        "instruction": question["instruction"],
                        "response": answer["code"],
                        "entry_point": question["entry_point"],
                        "skill": question["skill"],
                        "form": question["form"],
                        "level": question["level"],
                        "tests": answer["tests"],
                        "attempt": answer["attempt"],
                    }
                )
                break
            else:
                outcome = "failed"
        attempts = len(list((output / "answers").glob(f"{question['id']}_*.json")))
        if outcome == "failed" and attempts < MAX_ATTEMPTS:
            outcome = "failed, one retry left"
        counts[outcome] += 1
        counts[f"{group}: {outcome}"] += 1
    write_jsonl(output / "accepted.jsonl", accepted)
    spent = Budget(output, float("inf")).spent
    report = {
        "questions": len(questions),
        "accepted": len(accepted),
        "accounted_usd": round(spent, 4),
        "usd_per_accepted": round(spent / len(accepted), 5) if accepted else None,
        "counts": dict(sorted(counts.items())),
    }
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    print(f"Accepted rows: {output / 'accepted.jsonl'}")


def verify(args):
    output = args.output
    if not (output / "questions.jsonl").exists():
        raise RuntimeError("No questions.jsonl here. Run distill_questions.py first.")
    questions = {row["id"]: row for row in read_jsonl(output / "questions.jsonl")}
    todo = [
        path
        for path in sorted((output / "answers").glob("*.json"))
        if not (output / "verified" / path.name).exists()
        and load_json(path)["status"] == "unverified"
    ]
    print(f"{len(todo)} answers to test.")
    if todo:
        docker = shlex.split(args.docker)
        try:
            found = subprocess.run(
                [*docker, "image", "inspect", args.image],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker was not found on this machine.") from exc
        if found.returncode != 0:
            raise RuntimeError(
                f"Docker image missing or Docker not usable. Try: docker pull {args.image}"
            )
        command = docker_command(docker, args.image)

        def work(path):
            answer = load_json(path)
            result = judge(command, questions[answer["id"]], answer)
            save_json(output / "verified" / path.name, result)
            return f"{path.stem}: {'passed' if result['passed'] else result['reason']}"

        try:
            failed = run_parallel(work, todo, args.workers)
        finally:
            export(output)
        if failed:
            raise RuntimeError(
                f"{failed} answers could not be tested; repeat to retry."
            )
        print(
            "To retry the failures once: python distill_answers.py, then verify again."
        )
    else:
        export(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4, help="Containers at once")
    parser.add_argument("--docker", default="docker", help='For example "sudo docker"')
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive.")

    def stage():
        with locked(args.output):
            verify(args)

    return finish(stage)


if __name__ == "__main__":
    raise SystemExit(main())
