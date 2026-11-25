"""Grade benchmark_validation.py answers by running each row's tests in Docker.

Uses the same locked-down container and runner as curate_verify.py.

    python grade_validation.py runs/benchmarks/s2-ckpt6000-val
"""

import argparse
import json
import shlex
from collections import Counter
from pathlib import Path

from curate_collect import FENCE, clean_code
from curate_verify import BATCH, BATCH_RUNNER, run_batch
from distill_common import read_jsonl, run_parallel
from distill_verify import IMAGE, TIME_LIMIT, docker_command


def extract(answer):
    """The code to test: the first fenced block if any, minus example calls."""
    blocks = FENCE.findall(answer)
    code = blocks[0] if blocks else answer
    try:
        return clean_code(code)
    except ValueError:
        return code  # Out-of-scope or unusual code: test it as written.


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()

    rows = read_jsonl(args.folder / "answers.jsonl")
    for row in rows:
        row["response"] = extract(row["answer"])
    command = docker_command(
        shlex.split(args.docker), IMAGE, BATCH_RUNNER, BATCH * TIME_LIMIT + 60
    )
    results = {}

    def work(batch):
        passed = run_batch(command, batch)
        if passed is None:
            return "container failed; batch not graded"
        for row, ok in zip(batch, passed):
            results[row["id"]] = ok
        return f"{sum(passed)}/{len(batch)} passed"

    batches = [rows[start : start + BATCH] for start in range(0, len(rows), BATCH)]
    run_parallel(work, batches, args.workers)

    graded = [row for row in rows if row["id"] in results]
    by_source = Counter(row["source"] for row in graded)
    passed_by_source = Counter(row["source"] for row in graded if results[row["id"]])
    summary = {
        "graded": len(graded),
        "ungraded": len(rows) - len(graded),
        "pass_rate": round(sum(results.values()) / max(1, len(graded)), 4),
        "by_source": {
            source: f"{passed_by_source[source]}/{count}"
            for source, count in sorted(by_source.items())
        },
        "hit_limit": sum(row["hit_limit"] for row in rows),
    }
    with (args.folder / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in graded:
            file.write(
                json.dumps({"id": row["id"], "passed": results[row["id"]]}) + "\n"
            )
    (args.folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
