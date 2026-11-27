"""On-policy distillation: S3 writes answers, a bigger Qwen coder grades every token.

    python train_distill.py check   # pick the teacher prompt (no training)
    python train_distill.py         # train

Every step:
  1. take TASKS_PER_STEP prompts from PROMPTS (built by distill_pool.py and
     distill_filter.py: tasks the teacher solves, mostly ones S3 gets wrong),
  2. the student (S3) writes one sampled answer to each, at temperature 1,
  3. the teacher reads the same answer tokens after its own prompt and gives its
     probability for every possible next token at every answer position,
  4. one update that lowers the reverse KL between student and teacher at every
     answer token: sum over tokens v of p_student(v) * (log p_student(v) - log p_teacher(v)).

Unlike GRPO, every answer teaches something (no all-pass/all-fail groups) and
every token gets its own feedback. No tests are needed for training; they are
only used for the held-out check and the logged sample pass rate.

Reuses train_grpo.py's model, generation, tests, held-out check and saving
(importing it loads S3 and the tasks without training). Set SANDBOX there
("local" on a rented GPU).
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import train_grpo as grpo  # Student model, tokenizer, tasks, generate(), tests.

# ---- Settings
TEACHER = "models/qwen-coder-7b-instruct"  # Qwen/Qwen2.5-Coder-7B-Instruct.
OUTPUT = Path("runs/distill")
PROMPTS = "data/opd/prompts.jsonl"  # Read when training starts, not on import.
STEPS = 200
TASKS_PER_STEP = 64  # One answer per task: each answer gets its own feedback.
TEMPERATURE = 1.0  # The loss matches the untempered distribution, so sample from it.
# Runs 1-3 (1e-5, 5e-6, 5e-6) all drifted into answers that never end; run 3
# (raw-code-both teacher, Nov 26) into loops of repeated import lines from
# step 13, which the teacher scores as likely (KL fell 0.24 -> 0.10). Untested.
LEARNING_RATE = 1e-6
WARMUP_STEPS = 10
MAX_TOKENS = 8_000  # Padded student tokens per chunk (full-vocabulary scores are big).
SAVE_EVERY = 25  # Also how often the held-out check runs.
MAX_LIMIT_STEPS = 3  # Stop if this many steps in a row have over a quarter of answers at the token limit.
SEED = 0
# What the teacher sees before the student's answer. S3 answers with bare code
# (no Markdown fence, no explanation). The 7B teacher writes a ```python fence
# even when told not to (GPU check, Nov 26), so the "fenced" options start its
# reply with the fence themselves: it then scores S3's code as code. The
# student's end token, where a fenced teacher expects the closing fence, is left
# out of the loss. `python train_distill.py check` compares the options.
TEACHER_PROMPT = "raw-code-both"  # See IGNORED for why not "fenced".
FENCE = "```python\n"
RAW_CODE = (
    "You are a helpful assistant. Reply with only the Python code: "
    "no Markdown code fences and no explanation."
)
TEACHER_PROMPTS = {  # name: (system message, text added after the question, reply start)
    "default": (None, "", ""),  # The teacher's own default system message.
    "same-as-student": ("You are a helpful assistant.", "", ""),
    "raw-code-both": (
        RAW_CODE,
        "\n\nReply with only the Python code, without Markdown fences or explanation.",
        "",
    ),
    "fenced": (None, "", FENCE),
    "fenced-same-as-student": ("You are a helpful assistant.", "", FENCE),
}

grpo.OUTPUT = OUTPUT  # grpo.log() and checkpoints write here.
grpo.TEMPERATURE = TEMPERATURE  # Used by grpo.generate().
grpo.GENERATE_TASKS = TASKS_PER_STEP  # One generate() call per step.
student, device = grpo.model, grpo.device
optimizer = grpo.optimizer  # AdamW over the student, saved by grpo.save().

teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, local_files_only=True, dtype=torch.bfloat16
).to(device)
teacher.eval()
teacher.requires_grad_(False)
STOP_TOKENS = set(grpo.STOP_IDS)  # <|im_end|> and <|endoftext|>.
VOCAB = student.config.vocab_size  # 151,936; the teacher has extra unused rows.
# Tokens left out of the comparison: every token containing a Markdown fence
# (both distributions are renormalized over the rest). The raw-code teacher
# still puts weight on opening with ```; S3 never writes fences.
IGNORED = torch.zeros(VOCAB, dtype=torch.bool, device=device)
IGNORED[[i for i in range(VOCAB) if "```" in teacher_tokenizer.decode([i])]] = True
# Nov 26 history: run 1 used the "fenced" prompt with the end token dropped
# (LR 1e-5); run 2 also left the end tokens out of the KL (LR 5e-6). Both made
# S3 stop ending its answers (512-token answers with tests, a fence and prose
# by step 12-20; run 1 held-out 0.72 -> 0.00), because a fenced teacher wants
# more after the code: at the point where S3's greedy answers end it gave
# ending (or a fence) 4% probability. The raw-code-both teacher gave 79% (top
# token for 43 of 48), so the end token stays in the loss with that prompt.


def teacher_prompt_ids(task, prompt_name):
    system, suffix, reply_start = TEACHER_PROMPTS[prompt_name]
    messages = [{"role": "user", "content": task["instruction"] + suffix}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    text = teacher_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return teacher_tokenizer(text + reply_start, add_special_tokens=False)["input_ids"]


def make_examples(answers, prompt_name):
    """Both models' inputs for each answer; the answer tokens are identical.

    With a fenced teacher prompt the final end token is dropped: that teacher
    expects the closing fence there.
    """
    fenced = bool(TEACHER_PROMPTS[prompt_name][2])
    examples = []
    for a in answers:
        answer = a["answer_ids"]
        if fenced and answer and answer[-1] in STOP_TOKENS:
            answer = answer[:-1]
        if answer and not a["hit_limit"]:  # Unfinished (often looping) answers teach nothing good.
            examples.append(
                {
                    "student_ids": a["prompt_ids"] + answer,
                    "teacher_ids": teacher_prompt_ids(a["task"], prompt_name) + answer,
                    "answer_length": len(answer),
                }
            )
    return examples


def padded(rows):
    """Right-padded ids and attention mask (scoring, like SFT, pads on the right)."""
    longest = max(len(r) for r in rows)
    ids = torch.full((len(rows), longest), grpo.PAD)
    mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return ids.to(device), mask.to(device)


def answer_positions(chunk, key):
    """(row, position) of every logit that predicts an answer token, in order.

    Position i predicts token i + 1, so the answer's first token is predicted
    at the last prompt position. Prompts differ in length between the two
    models, so each model gets its own positions; the order of tokens is the same.
    """
    rows, positions = [], []
    for row, e in enumerate(chunk):
        start = len(e[key]) - e["answer_length"] - 1
        rows += [row] * e["answer_length"]
        positions += range(start, start + e["answer_length"])
    return torch.tensor(rows, device=device), torch.tensor(positions, device=device)


def chunk_kl(chunk, with_grad=True):
    """Reverse KL at each answer token of the chunk, shape [answer tokens]."""
    t_ids, t_mask = padded([e["teacher_ids"] for e in chunk])
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=grpo.USE_BF16):
        t_logits = teacher(input_ids=t_ids, attention_mask=t_mask, use_cache=False).logits
    t_rows, t_pos = answer_positions(chunk, "teacher_ids")
    # -1e4, not -inf: exp gives exactly 0 and nothing becomes NaN.
    t_logp = F.log_softmax(t_logits[t_rows, t_pos, :VOCAB].float().masked_fill(IGNORED, -1e4), dim=-1)
    del t_logits

    s_ids, s_mask = padded([e["student_ids"] for e in chunk])
    with torch.set_grad_enabled(with_grad), torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=grpo.USE_BF16
    ):
        s_logits = student(input_ids=s_ids, attention_mask=s_mask, use_cache=False).logits
        s_rows, s_pos = answer_positions(chunk, "student_ids")
        s_logp = F.log_softmax(s_logits[s_rows, s_pos].float().masked_fill(IGNORED, -1e4), dim=-1)
        return (s_logp.exp() * (s_logp - t_logp)).sum(-1)


def make_chunks(examples):
    """Sort by length and cap padded student tokens per chunk (as in train_grpo.py)."""
    examples = sorted(examples, key=lambda e: len(e["teacher_ids"]))
    chunks, chunk = [], []
    for e in examples:
        if chunk and (len(chunk) + 1) * len(e["teacher_ids"]) > MAX_TOKENS:
            chunks.append(chunk)
            chunk = []
        chunk.append(e)
    chunks.append(chunk)
    return chunks


def check(tasks=16):
    """Average KL per answer token, and on the first 3 tokens, for each teacher prompt."""
    torch.manual_seed(SEED)
    answers = grpo.generate(random.Random(SEED).sample(grpo.read_tasks(PROMPTS), tasks), 1, sample=True)
    print(f"{tasks} sampled S3 answers, median {sorted(len(a['answer_ids']) for a in answers)[tasks // 2]} tokens")
    for name in TEACHER_PROMPTS:
        kls = [chunk_kl([e], with_grad=False) for e in make_examples(answers, name)]
        every = torch.cat(kls)
        first = torch.cat([k[:3] for k in kls])
        print(f"{name:16} KL per token {every.mean():.3f}   first 3 tokens {first.mean():.3f}")
    # How the teacher answers on its own with the chosen prompt.
    ids = teacher_prompt_ids(answers[0]["task"], TEACHER_PROMPT)
    with torch.no_grad():
        out = teacher.generate(
            torch.tensor([ids], device=device), max_new_tokens=200, do_sample=False
        )
    print(f"\nTeacher's own answer with {TEACHER_PROMPT!r}:\n")
    print(teacher_tokenizer.decode(out[0, len(ids):], skip_special_tokens=True))


def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prompts = grpo.read_tasks(PROMPTS)
    print("Distillation prompts:", len(prompts))
    order = list(range(len(prompts)))
    random.Random(SEED).shuffle(order)
    student.train()
    grpo.log({"step": 0, "dev_pass_rate": grpo.dev_pass_rate()})
    newest_checkpoint, long_steps = None, 0
    for step in range(STEPS):
        started = time.time()
        learning_rate = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        tasks = [
            prompts[order[(step * TASKS_PER_STEP + i) % len(order)]]
            for i in range(TASKS_PER_STEP)
        ]
        torch.manual_seed(SEED * 100_000 + step)
        answers = grpo.generate(tasks, 1, sample=True)
        generated = time.time()

        examples = make_examples(answers, TEACHER_PROMPT)
        graded = sum(e["answer_length"] for e in examples)
        total_kl = 0.0
        for chunk in make_chunks(examples):
            kl = chunk_kl(chunk).sum()
            (kl / graded).backward()
            total_kl += kl.item()
        grad = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        if not math.isfinite(grad):
            raise FloatingPointError(f"Step {step + 1}: gradient is not finite.")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updated = time.time()

        done = step + 1
        with (OUTPUT / "samples.jsonl").open("a", encoding="utf-8") as file:  # The longest answer.
            longest = max(answers, key=lambda a: len(a["answer_ids"]))
            file.write(json.dumps({"step": done, "text": grpo.tokenizer.decode(longest["answer_ids"])}) + "\n")
        long_steps =long_steps + 1 if sum(a["hit_limit"] for a in answers) > len(answers) // 4 else 0
        grpo.log(
            {
                "step": f"{done}/{STEPS}",
                "kl": round(total_kl / graded, 4),  # Should fall.
                "sample_pass": round(sum(grpo.rewards_for(answers)) / len(answers), 3),
                "tokens": round(graded / len(examples)),
                "hit_limit": sum(a["hit_limit"] for a in answers),
                "grad": round(grad, 3),
                "lr": f"{learning_rate:.1e}",
                "seconds": f"{generated - started:.0f}+{updated - generated:.0f}",
            }
        )
        if done % SAVE_EVERY == 0 or done == STEPS:
            grpo.log({"step": done, "dev_pass_rate": grpo.dev_pass_rate()})
            folder = OUTPUT / (f"checkpoint-{done}" if done < STEPS else "final")
            grpo.save(folder, step=done)
            if newest_checkpoint:  # Only the newest keeps the optimizer state.
                (newest_checkpoint / "training_state.pt").unlink(missing_ok=True)
            newest_checkpoint = folder
        if long_steps >= MAX_LIMIT_STEPS:
            raise RuntimeError(f"Step {done}: answers keep running to the token limit; stopping.")
    print("Saved the distilled model to", OUTPUT / "final")


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else train()
