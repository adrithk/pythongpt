"""Small DeepSeek collector: topics -> briefs -> question/code -> saved dictionaries.

Rows are UNVERIFIED candidates. Run deepseek_pipeline.py for Docker verification
and training-dataset export. This file never executes the generated code.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-flash"
MAX_OUTPUT_TOKENS = 8192
BATCH_SIZE = 10
EOS = "<|im_end|>"
# Conservative peak prices, USD per million tokens; update if the provider changes them.
INPUT_PRICE, OUTPUT_PRICE = 0.30, 1.20
RULES = """Create original Python standard-library function tasks.
Use deterministic functions, JSON-compatible inputs/outputs, and no input mutation.
State the signature and edge cases. No files, network, classes or external packages.
Do not copy HumanEval, MBPP or other benchmark questions. Return JSON.
"""


class BudgetLimit(RuntimeError):
    pass


def save_json(path, value):
    """Write a whole file atomically, so an interrupted write cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def fingerprint(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def request_json(client, state, job, prompt, fields):
    """Reuse saved replies; allow at most two NEW calls per job per invocation."""
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": state["effort"],
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    folder = state["output"] / "requests" / job
    attempts = len(list(folder.glob("*.json")))
    for attempt in range(attempts + 2):
        path = folder / f"{attempt:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_hash"] != fingerprint(request):
                raise RuntimeError(
                    "Saved prompt/settings changed. Choose a new --output."
                )
        else:
            input_allowance = 2 * len(json.dumps(request).encode()) + 4096
            reserve = (
                input_allowance * INPUT_PRICE + MAX_OUTPUT_TOKENS * OUTPUT_PRICE
            ) / 1e6
            if state["spent"] + reserve > state["budget"]:
                raise BudgetLimit(
                    f"Budget reached; accounted USD {state['spent']:.4f}."
                )
            saved = {
                "request_hash": fingerprint(request),
                "request": request,
                "cost_usd": reserve,
                "response": None,
            }
            save_json(
                path, saved
            )  # Reserve BEFORE calling; ambiguous failures stay charged.
            state["spent"] += reserve
            try:
                response = client.chat.completions.create(**request).model_dump(
                    mode="json"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"DeepSeek request failed ({type(exc).__name__}); progress is saved."
                ) from exc
            usage = response.get("usage") or {}
            incoming, outgoing = (
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            cost = reserve
            if all(type(n) is int and n >= 0 for n in (incoming, outgoing)):
                cost = (incoming * INPUT_PRICE + outgoing * OUTPUT_PRICE) / 1e6
            saved.update(response=response, cost_usd=cost)
            save_json(path, saved)  # Includes raw text, usage and response ID.
            state["spent"] += cost - reserve
        try:
            response = saved["response"]
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                continue
            value = json.loads(choice["message"]["content"])
            if not isinstance(value, dict):
                continue
            if any(
                not isinstance(value.get(key), kind) or not value[key]
                for key, kind in fields.items()
            ):
                continue
            receipt = {
                "request_file": str(path.relative_to(state["output"])),
                "model": response.get("model"),
                "usage": response.get("usage"),
                "response_id": response.get("id"),
            }
            return value, receipt
        except (ValueError, TypeError, KeyError, IndexError):
            continue
    raise RuntimeError(
        f"{job}: no complete valid JSON after two new attempts. Replies are saved."
    )


def make_record(answer, seed, index, receipt, seen):
    """Validate syntax, reject exact repeats and build one candidate dictionary."""
    instruction, code, entry = (
        answer[key].strip() for key in ("instruction", "code", "entry_point")
    )
    if not instruction or not code or not entry.isidentifier():
        raise ValueError("Empty question/code or invalid function name")
    if any(
        token in text
        for token in ("<|im_start|>", EOS, "<|endoftext|>")
        for text in (instruction, code)
    ):
        raise ValueError("Unexpected chat control token")
    tree = ast.parse(code)  # Parses code; does NOT execute it.
    compile(tree, "<candidate>", "exec")
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == entry for node in tree.body
    ):
        raise ValueError("Requested function is missing")
    hashes = (
        fingerprint(" ".join(instruction.lower().split())),
        fingerprint(ast.dump(tree, include_attributes=False)),
    )
    if any(key in seen for key in hashes):
        raise ValueError("Exact repeated instruction or code")
    seen.update(hashes)
    return {
        "id": f"candidate_{index:06d}",
        "seed_id": f"seed_{index:06d}",
        **seed,
        "instruction": instruction,
        "response": code,
        "entry_point": entry,
        "status": "unverified",
        "hashes": list(hashes),
        "provenance": receipt,
    }


def export_candidates(output):
    """JSONL = one Python dictionary serialized as JSON per line."""
    temporary = output / "candidates.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as file:
        for path in sorted((output / "examples").glob("*.json")):
            file.write(
                json.dumps(json.loads(path.read_text()), ensure_ascii=False) + "\n"
            )
    temporary.replace(output / "candidates.jsonl")


def collect(client, args):
    """The main flow. Everything below is planning, generating and saving."""
    output = args.output
    if (output / "state.sqlite").exists():
        raise RuntimeError(
            "That is an advanced-pipeline directory. Choose a new --output."
        )
    settings = {
        "model": MODEL,
        "effort": args.effort,
        "rules": RULES,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "batch_size": BATCH_SIZE,
        "prices": [INPUT_PRICE, OUTPUT_PRICE],
    }
    config = output / "settings.json"
    if config.exists() and json.loads(config.read_text()) != settings:
        raise RuntimeError(
            "Settings changed. Use the original settings or a new --output."
        )
    save_json(config, settings)
    state = {
        "output": output,
        "effort": args.effort,
        "budget": args.budget_usd,
        "spent": sum(
            json.loads(p.read_text())["cost_usd"]
            for p in (output / "requests").glob("*/*.json")
        ),
    }
    seen = set()
    for path in (output / "examples").glob("*.json"):
        seen.update(json.loads(path.read_text()).get("hashes", []))
    try:
        plan, _ = request_json(
            client,
            state,
            "topics",
            RULES
            + """
Choose 20 diverse Python programming topics yourself: fundamentals, parsing,
collections, algorithms and useful less-common skills. No full problems yet.
Return {"topics":["topic 1","topic 2"]}, with 20 distinct nonempty strings.
""",
            {"topics": list},
        )
        topics = plan["topics"]
        if not 8 <= len(topics) <= 40 or not all(
            isinstance(t, str) and t.strip() for t in topics
        ):
            raise ValueError("Expected 8 to 40 nonempty topic strings")
        if len(set(topics)) != len(topics):
            raise ValueError("Repeated topics")
        for index in range(args.candidates):
            path = output / "examples" / f"{index:06d}.json"
            if path.exists():
                continue
            start = index // BATCH_SIZE * BATCH_SIZE
            batch_path = output / "briefs" / f"{start:06d}.json"
            if batch_path.exists():
                briefs = json.loads(batch_path.read_text())
            else:
                selected = [
                    topics[(start + j) % len(topics)] for j in range(BATCH_SIZE)
                ]
                recent = [
                    json.loads(p.read_text()).get("brief", "")
                    for p in sorted((output / "examples").glob("*.json"))[-20:]
                ]
                prompt = (
                    RULES
                    + f"""
Create {BATCH_SIZE} distinct problem briefs, one per topic in this ordered list:
{json.dumps(selected)}
Existing recent ideas to avoid paraphrasing: {json.dumps(recent)}
Each brief should specify a core operation, inputs/output and meaningful constraint.
Return exactly {BATCH_SIZE} strings: {{"briefs":["brief 1","brief 2"]}}.
"""
                )
                batch, _ = request_json(
                    client, state, f"briefs_{start:06d}", prompt, {"briefs": list}
                )
                briefs = batch["briefs"]
                if len(briefs) != BATCH_SIZE or not all(
                    isinstance(b, str) and b.strip() for b in briefs
                ):
                    raise ValueError("Wrong brief count or invalid brief")
                if len(set(briefs)) != len(briefs):
                    raise ValueError("Repeated briefs in one batch")
                save_json(batch_path, briefs)
            seed = {
                "topic": topics[index % len(topics)],
                "brief": briefs[index - start],
            }
            prompt = (
                RULES
                + "\n"
                + json.dumps(seed)
                + """
Expand this brief into a fully specified question and a concise correct solution.
The instruction must include the signature, edge cases and no-input-mutation rule,
and request code only. Put plain Python, without Markdown fences, in code.
Return {"instruction":"...", "code":"...", "entry_point":"function_name"}.
"""
            )
            answer, receipt = request_json(
                client,
                state,
                f"example_{index:06d}",
                prompt,
                {"instruction": str, "code": str, "entry_point": str},
            )
            try:
                record = make_record(answer, seed, index, receipt, seen)
            except (ValueError, SyntaxError) as exc:
                record = {
                    "id": f"candidate_{index:06d}",
                    **seed,
                    "status": "rejected",
                    "reason": str(exc),
                    "provenance": receipt,
                }
            save_json(path, record)
            print(
                f"{index + 1}/{args.candidates}: {record['status']}; accounted USD {state['spent']:.4f}",
                flush=True,
            )
    finally:
        export_candidates(output)  # Also rebuilds a usable JSONL after Ctrl+C.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["run"], default="run")
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Total candidate attempts, including saved rows",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        required=True,
        help="Cumulative accounting ceiling for this folder",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "deepseek-candidates"
    )
    parser.add_argument("--effort", choices=["low", "high"], default="high")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.candidates < 1
        or not math.isfinite(args.budget_usd)
        or args.budget_usd <= 0
    ):
        parser.error("Use a positive candidate count and a finite positive budget.")
    if args.dry_run:
        print(
            f"No API calls. Would collect {args.candidates} candidates in {args.output}."
        )
        return 0
    try:
        import fcntl

        from openai import OpenAI

        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            key = os.environ.get("DEEPSEEK_API_KEY") or getpass("DeepSeek API key: ")
            client = OpenAI(
                api_key=key,
                base_url="https://api.deepseek.com",
                max_retries=0,
                timeout=180,
            )
            collect(client, args)
        print(
            f"Saved {args.output / 'candidates.jsonl'} — unverified; test before training."
        )
        return 0
    except KeyboardInterrupt:
        print("Stopped. Repeat the command to resume saved work.")
        return 130
    except BudgetLimit as exc:
        print(f"{exc} Increase --budget-usd to continue the same folder.")
        return 2
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
