"""Shared plumbing for the three distillation stages.

    distill_questions.py  ->  distill_answers.py  ->  distill_verify.py

Nothing here knows about questions or answers. It only saves files, keeps the
spending ceiling, makes one API call with saved replies, and runs jobs in parallel.
"""

import ast
import contextlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "distill"
MODEL = "deepseek-flash"
MAX_OUTPUT_TOKENS = 16384  # Reasoning counts toward this allowance too.
# Peak prices, USD per million tokens, checked 14 Nov 2026. Off-peak is half.
INPUT_PRICE, OUTPUT_PRICE = 0.30, 1.20
CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


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


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    """JSONL = one dictionary per line."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_python(code):
    """Parse code WITHOUT running it; any syntax error becomes a ValueError."""
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"not valid Python: {exc.msg}") from exc


class Budget:
    """Running total of accounted spending for one output folder.

    Several threads add to it at once, so every change holds the lock.
    """

    def __init__(self, output, limit):
        self.limit = limit
        self.lock = threading.Lock()
        self.spent = sum(
            load_json(path)["cost_usd"]
            for path in (output / "requests").glob("*/*.json")
        )

    def add(self, amount, check=False):
        with self.lock:
            if check and self.spent + amount > self.limit:
                raise BudgetLimit(f"Budget reached; accounted USD {self.spent:.4f}.")
            self.spent += amount


@contextlib.contextmanager
def locked(output):
    """Allow one writer per output folder at a time."""
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another stage is already using {output}.") from exc
        yield


def make_client():
    from openai import OpenAI  # The OpenAI client library, pointed at DeepSeek.

    key = os.environ.get("DEEPSEEK_API_KEY") or getpass("DeepSeek API key: ")
    return OpenAI(
        api_key=key, base_url="https://api.deepseek.com", max_retries=0, timeout=600
    )


def parse_reply(response, fields):
    """Return the reply's JSON dictionary, or None if it is unusable."""
    try:
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":  # "length" = ran out of tokens.
            return None
        value = json.loads(choice["message"]["content"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None  # Includes the blank replies JSON mode sometimes returns.
    if not isinstance(value, dict):
        return None
    for key, kind in fields.items():
        if not isinstance(value.get(key), kind) or not value[key]:
            return None
    return value


def ask_json(client, budget, output, job, prompt, fields, effort):
    """Ask for one JSON object. Saved replies are reused for free.

    At most two NEW paid calls are made per job each time a stage runs.
    effort is "none" (no thinking), "low", "high" or "max".
    """
    request = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "extra_body": {
            "thinking": {"type": "disabled" if effort == "none" else "enabled"}
        },
    }
    if effort != "none":
        request["reasoning_effort"] = effort
    folder = output / "requests" / job
    already_saved = len(list(folder.glob("*.json")))
    for attempt in range(already_saved + 2):
        path = folder / f"{attempt:03d}.json"
        if path.exists():
            saved = load_json(path)
            if saved["request"] != request:
                continue  # An older attempt with a different prompt.
        else:
            # Worst case: every prompt byte is a token and the reply uses the
            # whole allowance. Reserve that BEFORE calling.
            reserve = (
                len(json.dumps(request).encode()) * INPUT_PRICE
                + MAX_OUTPUT_TOKENS * OUTPUT_PRICE
            ) / 1e6
            budget.add(reserve, check=True)
            saved = {"request": request, "cost_usd": reserve, "response": None}
            save_json(path, saved)
            try:
                reply = client.chat.completions.create(**request)
            except Exception as exc:  # The call may still have been billed.
                raise RuntimeError(
                    f"{job}: request failed ({type(exc).__name__}); reservation kept"
                ) from exc
            response = reply.model_dump(mode="json")
            usage = response.get("usage") or {}
            tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            cost = reserve
            if all(type(count) is int and count >= 0 for count in tokens):
                cost = (tokens[0] * INPUT_PRICE + tokens[1] * OUTPUT_PRICE) / 1e6
            saved.update(response=response, cost_usd=cost)
            save_json(path, saved)  # Raw reply, reasoning and usage are kept.
            budget.add(cost - reserve)  # Swap the reservation for the real cost.
        value = parse_reply(saved["response"], fields)
        if value is not None:
            receipt = {
                "request_file": str(path.relative_to(output)),
                "model": saved["response"].get("model"),
                "usage": saved["response"].get("usage"),
            }
            return value, receipt
    raise RuntimeError(f"{job}: no usable JSON reply after two new attempts")


def run_parallel(work, items, workers):
    """Call work(item) for every item on several threads; return the failure count.

    work returns a short progress message. One failed item does not stop the
    others. Reaching the budget stops everything that has not started yet.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(work, item) for item in items]
    failed = 0
    try:
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                message = future.result()
            except BudgetLimit:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop the rest
                failed += 1
                message = f"FAILED: {exc}"
            print(f"[{done}/{len(futures)}] {message}", flush=True)
    finally:
        # Requests already sent are allowed to finish so their replies are saved.
        pool.shutdown(wait=True, cancel_futures=True)
    return failed


def finish(stage):
    """Run a stage and turn the usual stops into one-line messages and exit codes."""
    try:
        stage()
        return 0
    except KeyboardInterrupt:
        print("Stopped. Repeat the command to resume saved work.")
        return 130
    except BudgetLimit as exc:
        print(f"{exc} Raise --budget-usd to continue in the same folder.")
        return 2
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"Stopped: {exc}")
        return 1
