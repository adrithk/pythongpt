"""Score the distillation prompts with the teacher and S3, then pick the set.

    python distill_filter.py          # on the GPU: score data/opd/candidates.jsonl
    python distill_filter.py select   # write data/opd/prompts.jsonl from the scores

Scoring: the teacher (with train_distill.py's TEACHER_PROMPT) and S3 each write
one greedy answer per candidate, and both answers run against the task's tests.
Writes data/opd/scores.jsonl (id, teacher_pass, s3_pass, token counts). Safe to
stop and rerun: scored ids are skipped.

Selecting keeps only tasks the teacher solves, so its token scores point toward
working code (a task whose tests the teacher fails is often a broken task).
Among those, every task S3 gets wrong is kept, and tasks S3 already solves fill
the rest up to SELECT_TOTAL: they still teach (the teacher's token choices on
code S3 writes correctly) but less.
"""

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

CANDIDATES = Path("data/opd/candidates.jsonl")
SCORES = Path("data/opd/scores.jsonl")
PROMPTS = Path("data/opd/prompts.jsonl")
TASKS_AT_ONCE = 256
SELECT_TOTAL = 12_800  # 200 steps x 64 prompts: one pass in train_distill.py.
SEED = 0


def read(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def score():
    import torch

    import train_distill as td  # Loads S3 (student) and the teacher.

    grpo = td.grpo
    grpo.GENERATE_TASKS = TASKS_AT_ONCE
    tok = td.teacher_tokenizer
    tok.padding_side = "left"
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]

    @torch.inference_mode()
    def teacher_answers(tasks):
        prompts = [td.teacher_prompt_ids(task, td.TEACHER_PROMPT) for task in tasks]
        longest = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), longest), tok.pad_token_id)
        mask = torch.zeros((len(prompts), longest), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, longest - len(p) :] = torch.tensor(p)
            mask[i, longest - len(p) :] = 1
        out = td.teacher.generate(
            input_ids=ids.to(td.device),
            attention_mask=mask.to(td.device),
            max_new_tokens=grpo.MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[0],
        )
        answers = []
        for task, new in zip(tasks, out[:, longest:].tolist()):
            stop = next((i for i, t in enumerate(new) if t in stop_ids), None)
            new = new if stop is None else new[:stop]
            # A fenced teacher prompt already opened the code block.
            text = td.TEACHER_PROMPTS[td.TEACHER_PROMPT][2] + tok.decode(new, skip_special_tokens=True)
            answers.append({"task": task, "text": text, "tokens": len(new)})
        return answers

    done = {row["id"] for row in read(SCORES)} if SCORES.exists() else set()
    todo = [t for t in read(CANDIDATES) if t["id"] not in done]
    print(f"Already scored: {len(done)}, to score: {len(todo)}", flush=True)
    with SCORES.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), TASKS_AT_ONCE):
            began = time.time()
            tasks = todo[start : start + TASKS_AT_ONCE]
            student = grpo.generate(tasks, 1, sample=False)
            teacher = teacher_answers(tasks)
            generated = time.time()
            s_pass = grpo.rewards_for(student)
            t_pass = grpo.rewards_for(teacher)
            for task, s, t, sp, tp in zip(tasks, student, teacher, s_pass, t_pass):
                record = {
                    "id": task["id"],
                    "teacher_pass": int(tp),
                    "s3_pass": int(sp),
                    "teacher_tokens": t["tokens"],
                    "s3_tokens": len(s["answer_ids"]),
                }
                out.write(json.dumps(record) + "\n")
            out.flush()
            print(
                f"{len(done) + start + len(tasks)}/{len(done) + len(todo)} | "
                f"teacher {sum(t_pass) / len(tasks):.2f} s3 {sum(s_pass) / len(tasks):.2f} | "
                f"{generated - began:.0f}+{time.time() - generated:.0f}s",
                flush=True,
            )


def select():
    tasks = {t["id"]: t for t in read(CANDIDATES)}
    scores = {row["id"]: row for row in read(SCORES)}
    teacher_ok = [i for i in tasks if scores.get(i, {}).get("teacher_pass")]
    s3_wrong = [i for i in teacher_ok if not scores[i]["s3_pass"]]
    s3_right = [i for i in teacher_ok if scores[i]["s3_pass"]]
    rng = random.Random(SEED)
    rng.shuffle(s3_right)
    chosen = s3_wrong + s3_right[: max(0, SELECT_TOTAL - len(s3_wrong))]
    rng.shuffle(chosen)
    with PROMPTS.open("w", encoding="utf-8") as file:
        for i in chosen:
            file.write(json.dumps(tasks[i], ensure_ascii=False) + "\n")

    def kinds(ids):
        return dict(Counter(f"{tasks[i]['style']}/{tasks[i]['difficulty']}" for i in ids))

    report = {
        "scored": len(scores),
        "teacher_pass": len(teacher_ok),
        "s3_pass": sum(r["s3_pass"] for r in scores.values()),
        "teacher_pass_s3_fail": len(s3_wrong),
        "chosen": len(chosen),
        "chosen_s3_fail": sum(1 for i in chosen if not scores[i]["s3_pass"]),
        "pass_rates_by_kind": {
            kind: {
                "n": n,
                "teacher": round(sum(scores[i]["teacher_pass"] for i in ids) / n, 3),
                "s3": round(sum(scores[i]["s3_pass"] for i in ids) / n, 3),
            }
            for kind, ids in _group(tasks, scores).items()
            for n in [len(ids)]
        },
        "chosen_by_kind": kinds(chosen),
    }
    Path("data/opd/prompts-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def _group(tasks, scores):
    groups = {}
    for i in scores:
        if i in tasks:
            kind = f"{tasks[i]['style']}/{tasks[i]['difficulty']}"
            groups.setdefault(kind, []).append(i)
    return groups


if __name__ == "__main__":
    select() if sys.argv[1:] == ["select"] else score()
