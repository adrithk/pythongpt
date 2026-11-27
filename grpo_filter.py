"""Keep only the GRPO tasks the starting model solves sometimes, not always.

    python grpo_filter.py

A task whose GROUP answers all pass, or all fail, teaches GRPO nothing (its
advantages are all 0), so in the first run about half of each step was wasted.
This lets the model in train_grpo.py's BASE (S3) write GROUP sampled answers
to every pool task, exactly as training would, and writes:

    data/grpo/pool-scores.jsonl  one line per task: id, source, passes (0..GROUP)
    data/grpo/pool-mixed.jsonl   the tasks with 0 < passes < GROUP

Uses train_grpo.py's settings (BASE, GROUP, TEMPERATURE, SANDBOX, ...); set
SANDBOX = "local" there on a rented GPU. Safe to stop and rerun: tasks already
in pool-scores.jsonl are skipped.
"""

import json
from collections import Counter
from pathlib import Path

import torch

import train_grpo as grpo  # Loads the model and the tasks; does not train.

SCORES = Path("data/grpo/pool-scores.jsonl")
MIXED = Path("data/grpo/pool-mixed.jsonl")
TASKS_AT_ONCE = 64  # Tasks generated and tested per round.
LIMIT = None  # A small number to score only the first tasks (quick tests).
# One generate() call per round (64 x GROUP = 512 answers at once): a small
# model leaves a big GPU mostly idle, so larger batches cost little extra time.
grpo.GENERATE_TASKS = TASKS_AT_ONCE

done = set()
if SCORES.exists():
    with SCORES.open(encoding="utf-8") as file:
        done = {json.loads(line)["id"] for line in file}
todo = [task for task in grpo.pool[:LIMIT] if task["id"] not in done]
print(f"Already scored: {len(done)}, to score: {len(todo)}")

torch.manual_seed(grpo.SEED)
with SCORES.open("a", encoding="utf-8") as scores:
    for start in range(0, len(todo), TASKS_AT_ONCE):
        tasks = todo[start : start + TASKS_AT_ONCE]
        rewards = grpo.rewards_for(grpo.generate(tasks, grpo.GROUP, sample=True))
        for i, task in enumerate(tasks):
            passes = int(sum(rewards[i * grpo.GROUP : (i + 1) * grpo.GROUP]))
            record = {"id": task["id"], "source": task["source"], "passes": passes}
            scores.write(json.dumps(record) + "\n")
        scores.flush()
        print(f"{len(done) + start + len(tasks)}/{len(done) + len(todo)} scored", flush=True)

# Rebuild the mixed file from all scores, in pool order.
with SCORES.open(encoding="utf-8") as file:
    passes = {row["id"]: row["passes"] for row in map(json.loads, file)}
mixed = [t for t in grpo.pool if 0 < passes.get(t["id"], 0) < grpo.GROUP]
with MIXED.open("w", encoding="utf-8") as file:
    for task in mixed:
        file.write(json.dumps(task, ensure_ascii=False) + "\n")

counts = Counter(passes.values())
print("Tasks by number of passing answers:", dict(sorted(counts.items())))
print(f"Mixed: {len(mixed)} of {len(passes)} -> {MIXED}")
print("Mixed by source:", dict(Counter(t["source"] for t in mixed)))
