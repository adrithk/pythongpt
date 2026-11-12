import json
import platform
from importlib.metadata import version
from pathlib import Path

from evalplus.data import get_human_eval_plus_hash
from evalplus.evaluate import evaluate


def main():
    folder = Path("/app")

    metadata = json.loads((folder / "metadata.json").read_text())

    grading_hash = get_human_eval_plus_hash()

    if metadata["dataset_hash"] != grading_hash:
        raise RuntimeError("Generation and grading datasets differ.")

    grader = {
        "evalplus": version("evalplus"),
        "python": platform.python_version(),
        "dataset_hash": grading_hash,
        "parallel": 1,
    }

    (folder / "grader.json").write_text(json.dumps(grader, indent=2))

    evaluate(
        dataset="humaneval",
        samples=str(folder / "samples.jsonl"),
        parallel=1,
    )


if __name__ == "__main__":
    main()
