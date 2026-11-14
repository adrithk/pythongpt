"""Curation step 2 of 3: re-run every row's tests on this machine.

The sources say their answers pass their tests. This step checks that here,
in Docker, with only the standard library installed. A row is kept only if its
code loads and every test passes. Rows that have no tests are kept if the code
at least loads without error.

Input: data/curated/candidates/<source>.jsonl
Output: data/curated/verified/<source>.jsonl
"""

import argparse
import json
import secrets
import shlex
import subprocess
import threading
from collections import Counter
from pathlib import Path

from curate_collect import OUTPUT, SOURCES
from distill_common import finish, locked, read_jsonl, run_parallel, write_jsonl
from distill_verify import IMAGE, RUNNER, TIME_LIMIT, docker_command, run_tests

BATCH = 100  # Rows tested per container. Starting a container per row is slow.

# This program runs INSIDE the container. It starts the one-row runner from
# distill_verify.py once per row, each as its own process with its own time
# limit, so one stuck or crashing row cannot spoil the rest of the batch.
BATCH_RUNNER = r"""
import json, subprocess, sys
batch = json.load(sys.stdin)
results = []
for job in batch["jobs"]:
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", batch["runner"]],
            input=json.dumps(job), capture_output=True, text=True, timeout=batch["limit"],
        )
        lines = [l for l in done.stdout.splitlines() if l.startswith(job["marker"])]
        results.append(json.loads(lines[-1][len(job["marker"]):]) if lines else None)
    except subprocess.TimeoutExpired:
        results.append(None)
sys.stdout.write(batch["marker"] + json.dumps(results) + "\n")
"""


def run_batch(command, rows, limit=TIME_LIMIT):
    """Test many rows in one container. Returns one True/False per row.

    If the whole container dies (for example it runs out of memory), the answer
    is None and the caller falls back to testing the rows one at a time.
    """
    marker = secrets.token_hex(8)
    jobs = [
        {"code": row["response"], "tests": row["tests"] or ["pass"], "marker": marker}
        for row in rows
    ]
    payload = {"jobs": jobs, "runner": RUNNER, "limit": limit, "marker": marker}
    try:
        done = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=len(rows) * limit + 120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(marker):
            results = json.loads(line[len(marker) :])
            return [errors is not None and not any(errors) for errors in results]
    return None


def verify(args):
    output = args.output
    candidates = output / "candidates" / f"{args.source}.jsonl"
    if not candidates.exists():
        raise RuntimeError(
            f"{candidates} not found. Run curate_collect.py {args.source} first."
        )
    rows = read_jsonl(candidates)
    checked_path = output / "verified" / f"{args.source}.checked.jsonl"
    checked_path.parent.mkdir(parents=True, exist_ok=True)
    checked = {}
    if checked_path.exists():  # Results from an earlier, interrupted run.
        checked = {row["id"]: row["passed"] for row in read_jsonl(checked_path)}
    todo = [row for row in rows if row["id"] not in checked]
    print(f"{len(todo)} of {len(rows)} rows still to test.")
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
            raise RuntimeError(f"Docker image missing. Try: docker pull {args.image}")
        many = docker_command(docker, args.image, BATCH_RUNNER, BATCH * TIME_LIMIT + 60)
        single = docker_command(docker, args.image)
        lock = threading.Lock()

        def work(batch):
            passed = run_batch(many, batch)
            if passed is None:  # Slow path: find the row that broke the container.
                passed = []
                for row in batch:
                    errors = run_tests(
                        single, row["response"], row["tests"] or ["pass"]
                    )
                    passed.append(errors is not None and not any(errors))
            with lock, checked_path.open("a", encoding="utf-8") as file:
                for row, ok in zip(batch, passed):
                    checked[row["id"]] = ok
                    file.write(json.dumps({"id": row["id"], "passed": ok}) + "\n")
            return f"{sum(passed)}/{len(batch)} passed"

        batches = [todo[start : start + BATCH] for start in range(0, len(todo), BATCH)]
        failed = run_parallel(work, batches, args.workers)
        if failed:
            raise RuntimeError(
                f"{failed} batches could not be tested; repeat to retry."
            )
    kept = [row for row in rows if checked.get(row["id"])]
    write_jsonl(output / "verified" / f"{args.source}.jsonl", kept)
    counts = Counter("passed" if checked.get(row["id"]) else "failed" for row in rows)
    print(
        f"{dict(counts)}; kept rows in {output / 'verified' / (args.source + '.jsonl')}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", choices=list(SOURCES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4, help="Containers at once")
    parser.add_argument("--docker", default="docker", help='For example "sudo docker"')
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args(argv)

    def stage():
        with locked(args.output):
            verify(args)

    return finish(stage)


if __name__ == "__main__":
    raise SystemExit(main())
