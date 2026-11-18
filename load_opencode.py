"""Download nvidia/OpenCodeInstruct once and load it from your own disk.

The dataset is 50 Parquet files (about 6.9 GB, about 5 million rows). Step 1
copies them into data/opencode-raw; step 2 opens them with `datasets`, which
maps the files from disk instead of reading them all into memory.

    python load_opencode.py --shards 1     # one file (~150 MB) to try it out
    python load_opencode.py                # all 50 files

Other scripts can reuse the loaded table:

    from load_opencode import load
    rows = load()          # a datasets.Dataset
    print(rows[0]["input"])
"""

import argparse
from collections import Counter
from pathlib import Path

DATASET = "nvidia/OpenCodeInstruct"
# Pinned so reruns get exactly the same rows.
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
RAW = Path(__file__).resolve().parent / "data" / "opencode-raw"
TOTAL_SHARDS = 50


def download(shards=TOTAL_SHARDS):
    """Copy the first `shards` Parquet files to RAW. Files already there are skipped."""
    from huggingface_hub import snapshot_download

    names = "data/train-{:05d}-of-" + f"{TOTAL_SHARDS:05d}.parquet"
    wanted = [names.format(i) for i in range(shards)]
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=wanted + ["README.md"],
        local_dir=RAW,
    )


def downloaded():
    """The Parquet files already in RAW, in order."""
    return sorted(str(path) for path in (RAW / "data").glob("train-*.parquet"))


def load():
    """Open every Parquet file already in RAW as one datasets.Dataset."""
    from datasets import load_dataset

    files = downloaded()
    if not files:
        raise FileNotFoundError(f"No Parquet files in {RAW / 'data'}. Download first.")
    return load_dataset("parquet", data_files=files, split="train")


def summarize(rows):
    """Print the size, the columns, a few counts and one example row."""
    print(f"{len(rows):,} rows")
    print("columns:", rows.column_names)
    for column in ("domain", "generation_algorithm"):
        print(column, dict(Counter(rows[column]).most_common()))
    passed = sum(float(score) == 1.0 for score in rows["average_test_score"])
    print(f"average_test_score == 1: {passed:,} of {len(rows):,}")
    example = rows[0]
    print("\n--- first row ---")
    for column in rows.column_names:
        print(f"[{column}]\n{str(example[column])[:600]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--shards",
        type=int,
        default=TOTAL_SHARDS,
        help=f"How many of the {TOTAL_SHARDS} files to download (default: all)",
    )
    args = parser.parse_args()
    download(min(args.shards, TOTAL_SHARDS))
    summarize(load())


if __name__ == "__main__":
    main()
