"""Grade benchmark_lcb.py answers against LiveCodeBench's hidden tests, in Docker.

    python grade_lcb.py runs/benchmarks/s3-lcb

A problem passes only if every public and private test passes (6 s per test).
Writes results.jsonl and summary.json (pass rate by platform and difficulty).
Comparison follows LiveCodeBench's rules in simplified form: whole-line text
match for stdin programs (numbers compared with a small tolerance), equal JSON
values for LeetCode methods.
"""

import base64
import io
import json
import pickle
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

from benchmark_lcb import DATA

IMAGE = "python:3.12-slim"
TIME_LIMIT = 6
PRELUDE = (
    "import sys, math, heapq, bisect, itertools, functools, collections, string, re\n"
    "from typing import *\nfrom collections import *\nfrom heapq import *\n"
    "from math import *\nfrom itertools import *\nfrom bisect import *\n"
    "from functools import *\nsys.setrecursionlimit(10**6)\n"
)

# Runs inside the container: one subprocess per test, stop at the first failure.
RUNNER = r'''
import json, subprocess, sys, tempfile, os
from concurrent.futures import ThreadPoolExecutor
TIME_LIMIT, PRELUDE = json.loads(sys.argv[1])
DRIVER = """
import json as _json, sys as _sys
_args = [_json.loads(x) for x in _sys.stdin.read().split("\\n") if x.strip()]
print(_json.dumps(getattr(Solution(), {name!r})(*_args)))
"""

def same(got, want, functional):
    if functional:
        try:
            return json.loads(got) == json.loads(want)
        except Exception:
            return False
    a = [l.strip() for l in got.strip().splitlines()]
    b = [l.strip() for l in want.strip().splitlines()]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty):
            return False
        for p, q in zip(tx, ty):
            if p == q:
                continue
            try:
                if abs(float(p) - float(q)) > 1e-6 * max(1.0, abs(float(q))):
                    return False
            except ValueError:
                return False
    return True

def grade(job):
    functional = job["func_name"] is not None
    program = PRELUDE + job["code"]
    if functional:
        program += "\n" + DRIVER.format(name=job["func_name"])
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "main.py")
        open(path, "w").write(program)
        for number, test in enumerate(job["tests"]):
            try:
                done = subprocess.run([sys.executable, path], input=test["input"],
                    capture_output=True, text=True, timeout=TIME_LIMIT, cwd=folder)
            except subprocess.TimeoutExpired:
                return {"passed": False, "reason": "timeout", "test": number}
            if done.returncode != 0:
                error = (done.stderr.strip().splitlines() or ["?"])[-1][:200]
                return {"passed": False, "reason": "error: " + error, "test": number}
            if not same(done.stdout, test["output"], functional):
                return {"passed": False, "reason": "wrong answer", "test": number}
    return {"passed": True, "reason": "", "test": len(job["tests"])}

jobs = json.load(open("/work/jobs.json"))
with ThreadPoolExecutor(4) as pool:
    results = list(pool.map(grade, jobs))
json.dump(results, open("/work/results.json", "w"))
'''


class NoCode(pickle.Unpickler):
    """Unpickle plain data only: refuse anything that would import or run code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def private_tests(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(text))
        return json.loads(NoCode(io.BytesIO(raw)).load())


def main():
    folder = Path(sys.argv[1]).resolve()
    problems = {}
    for line in DATA.open():
        row = json.loads(line)
        problems[row["question_id"]] = row
    answers = [json.loads(line) for line in (folder / "answers.jsonl").open()]
    jobs = []
    for answer in answers:
        row = problems[answer["question_id"]]
        tests = json.loads(row["public_test_cases"]) + private_tests(
            row["private_test_cases"]
        )
        func_name = json.loads(row["metadata"]).get("func_name")
        jobs.append(
            {"code": answer["code"], "tests": tests, "func_name": func_name}
        )
    (folder / "jobs.json").write_text(json.dumps(jobs))
    (folder / "runner.py").write_text(RUNNER)
    subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", "4", "--memory", "8g", "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,src={folder},dst=/work",
            IMAGE, "python", "/work/runner.py", json.dumps([TIME_LIMIT, PRELUDE]),
        ],
        check=True,
    )
    results = json.loads((folder / "results.json").read_text())
    (folder / "jobs.json").unlink()  # Large; the tests stay in the dataset file.
    (folder / "runner.py").unlink()
    (folder / "results.json").unlink()

    totals, passed = Counter(), Counter()
    with (folder / "results.jsonl").open("w") as file:
        for answer, result in zip(answers, results):
            file.write(json.dumps({"question_id": answer["question_id"], **result}) + "\n")
            for key in ("all", answer["platform"], answer["difficulty"],
                        f"{answer['platform']} {answer['difficulty']}"):
                totals[key] += 1
                passed[key] += result["passed"]
    summary = {
        key: f"{passed[key]}/{totals[key]} ({100 * passed[key] / totals[key]:.1f}%)"
        for key in sorted(totals)
    }
    summary["reasons"] = dict(Counter(r["reason"].split(":")[0] or "pass" for r in results))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
