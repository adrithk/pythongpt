"""Advanced DeepSeek dataset verification, generation, export and upload.

Importing this file does not call APIs. See DEEPSEEK_PIPELINE.md for commands.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "1"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "deepseek-distill"
DEFAULT_TOKENIZER = ROOT / "models" / "qwen-base"
BASE_URL = "https://api.deepseek.com"
EOS = "<|im_end|>"
ALLOWED_IMPORTS = {
    "bisect",
    "calendar",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
    "unicodedata",
}
BLOCKED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
    "help",
    "breakpoint",
    "type",
    "object",
    "super",
    "print",
    "BaseException",
    "SystemExit",
    "GeneratorExit",
}


class PipelineError(RuntimeError):
    pass


class InvalidCandidate(ValueError):
    pass


class BudgetReached(PipelineError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value: Any) -> str:
    text = value if isinstance(value, str) else dumps(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(dumps(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_object(value: Any) -> dict:
    if not isinstance(value, dict):
        raise InvalidCandidate("Expected a JSON object")
    return value


def text_field(row: dict, name: str, maximum: int = 20000) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise InvalidCandidate(
            f"{name}: expected a nonempty string <= {maximum} characters"
        )
    if any(t in value for t in ("<|im_start|>", EOS, "<|endoftext|>")):
        raise InvalidCandidate(f"{name} contains chat control tokens")
    return value.strip()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(left: str, right: str) -> float:
    a, b = words(left), words(right)
    return len(a & b) / max(1, len(a | b))


def code_hash(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body.pop(0)
    return digest(ast.dump(tree, include_attributes=False))


def check_code(code: str, entry_point: str) -> None:
    """Scope screening, not a security boundary. Execution requires Docker."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", entry_point):
        raise InvalidCandidate("Expected a lowercase Python function name")
    if not isinstance(code, str) or len(code) > 64000 or chr(96) * 3 in code:
        raise InvalidCandidate("Expected plain Python <= 64000 characters")
    try:
        tree = ast.parse(code)
        compile(tree, "<generated>", "exec")
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise InvalidCandidate(f"Invalid Python: {type(exc).__name__}") from exc
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == entry_point for n in tree.body
    ):
        raise InvalidCandidate(f"Missing top-level function {entry_point}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
            raise InvalidCandidate("Only synchronous function tasks are supported")
        if isinstance(node, ast.Name) and (
            node.id in BLOCKED_NAMES or node.id.startswith("__")
        ):
            raise InvalidCandidate(f"Out-of-scope name: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            raise InvalidCandidate(
                "Introspection/attribute reassignment is out of scope"
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if getattr(node, "level", 0) or any(
                n.split(".")[0] not in ALLOWED_IMPORTS for n in names
            ):
                raise InvalidCandidate(
                    "Import outside the supported standard-library subset"
                )
            if any(
                a.name.startswith("__") or (a.asname or "").startswith("__")
                for a in node.names
            ):
                raise InvalidCandidate("Reserved import name")


@dataclass(frozen=True)
class Settings:
    model: str = "deepseek-flash"
    effort: str = "high"
    max_output_tokens: int = 8192
    max_example_tokens: int = 2048
    api_attempts: int = 3
    repairs: int = 1
    seed_batch_size: int = 12
    min_tests: int = 8
    fuzz_cases: int = 32
    sandbox_timeout: int = 30
    input_price: float = 0.30
    output_price: float = 1.20
    split_seed: int = 42
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    image_id: str = ""
    tokenizer_hash: str = ""
    pipeline_version: str = PIPELINE_VERSION


@contextlib.contextmanager
def writer_lock(root: Path):
    try:
        import fcntl
    except ImportError as exc:
        raise PipelineError("Use WSL, Linux, or macOS") from exc
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a+") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(
                "Another process is writing to this output directory"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


class Store:
    """SQLite jobs, records and indexed lexical retrieval."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "state.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                job TEXT, attempt INTEGER, request_hash TEXT, status TEXT,
                cost REAL, raw_path TEXT, error TEXT, created TEXT,
                PRIMARY KEY (job, attempt));
            CREATE TABLE IF NOT EXISTS seeds (
                id TEXT PRIMARY KEY, category TEXT, brief TEXT, payload TEXT,
                state TEXT DEFAULT 'pending');
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, status TEXT, record TEXT);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, instruction TEXT, code_hash TEXT,
                text_hash TEXT, family_id TEXT, category TEXT);
            CREATE INDEX IF NOT EXISTS document_code ON documents(code_hash);
            CREATE INDEX IF NOT EXISTS document_text ON documents(text_hash);
            CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY CHECK(id=1), total REAL);
            INSERT OR IGNORE INTO ledger SELECT 1,COALESCE(SUM(cost),0) FROM calls;
            CREATE TRIGGER IF NOT EXISTS charge_insert AFTER INSERT ON calls
                BEGIN UPDATE ledger SET total=total+NEW.cost WHERE id=1; END;
            CREATE TRIGGER IF NOT EXISTS charge_update AFTER UPDATE OF cost ON calls
                BEGIN UPDATE ledger SET total=total+NEW.cost-OLD.cost WHERE id=1; END;
        """)
        try:
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(doc_id UNINDEXED, body)"
            )
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()

    def close(self):
        self.db.close()

    def charge(self) -> float:
        return max(
            0.0,
            float(self.db.execute("SELECT total FROM ledger WHERE id=1").fetchone()[0]),
        )

    def candidate(self, key: str) -> dict | None:
        row = self.db.execute(
            "SELECT record FROM candidates WHERE id=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_candidate(self, record: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?,?,?)",
            (record["id"], record["status"], dumps(record)),
        )
        if record["status"] in {"accepted", "rejected"}:
            self.db.execute(
                "UPDATE seeds SET state='done' WHERE id=?", (record["seed_id"],)
            )
        self.db.commit()

    def add_document(self, record: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?)",
            (
                record["id"],
                record["instruction"],
                code_hash(record["response"]),
                digest(normalized(record["instruction"])),
                record["family_id"],
                record["category"],
            ),
        )
        if self.fts:
            self.db.execute("DELETE FROM search WHERE doc_id=?", (record["id"],))
            self.db.execute(
                "INSERT INTO search VALUES (?,?)", (record["id"], record["instruction"])
            )
        self.db.commit()

    def rebuild_documents(self):
        for row in self.db.execute(
            "SELECT c.record FROM candidates c LEFT JOIN documents d ON d.id=c.id "
            "WHERE c.status='accepted' AND d.id IS NULL"
        ).fetchall():
            self.add_document(json.loads(row[0]))

    def exact_duplicate(self, instruction: str, code: str = "") -> str | None:
        row = self.db.execute(
            "SELECT id FROM documents WHERE text_hash=? OR code_hash=? LIMIT 1",
            (digest(normalized(instruction)), code_hash(code) if code else ""),
        ).fetchone()
        return row[0] if row else None

    def nearest(self, instruction: str, limit: int = 6) -> list[dict]:
        terms = sorted(
            (w for w in words(instruction) if len(w) > 3), key=lambda w: (-len(w), w)
        )[:20]
        if not terms:
            return []
        if self.fts:
            query = " OR ".join('"' + t + '"' for t in terms)
            rows = self.db.execute(
                "SELECT d.* FROM search s JOIN documents d ON d.id=s.doc_id "
                "WHERE search MATCH ? ORDER BY bm25(search) LIMIT ?",
                (query, max(40, limit)),
            ).fetchall()
        else:
            condition = " OR ".join("instruction LIKE ?" for _ in terms)
            rows = self.db.execute(
                f"SELECT * FROM documents WHERE {condition} LIMIT 100",
                tuple("%" + t + "%" for t in terms),
            ).fetchall()
        return sorted(
            (dict(r) for r in rows),
            key=lambda r: (-similarity(instruction, r["instruction"]), r["id"]),
        )[:limit]

    def rows(self, status: str = "accepted") -> list[dict]:
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT record FROM candidates WHERE status=? ORDER BY id", (status,)
            )
        ]

    def stats(self) -> dict:
        return {
            "candidates": dict(
                self.db.execute(
                    "SELECT status,COUNT(*) FROM candidates GROUP BY status"
                )
            ),
            "seeds": self.db.execute("SELECT COUNT(*) FROM seeds").fetchone()[0],
            "api_attempts": self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
            "conservative_accounted_usd": round(self.charge(), 6),
            "uncertain_calls": self.db.execute(
                "SELECT COUNT(*) FROM calls WHERE status IN ('requested','uncertain')"
            ).fetchone()[0],
        }


class Teacher:
    """Cached sequential requests and conservative cost reservations."""

    def __init__(self, store: Store, settings: Settings, budget: float, client=None):
        self.store, self.settings, self.budget, self.client = (
            store,
            settings,
            budget,
            client,
        )

    def _client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise PipelineError(
                    "Install requirements-distill.txt in your Python environment"
                ) from exc
            key = os.environ.get("DEEPSEEK_API_KEY") or getpass("DeepSeek API key: ")
            if not key.strip():
                raise PipelineError("A DeepSeek API key is required")
            self.client = OpenAI(
                api_key=key, base_url=BASE_URL, max_retries=0, timeout=180
            )
        return self.client

    def _cost(self, raw: dict, reserve: float) -> float:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return reserve
        inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if any(type(x) is not int or x < 0 for x in (inp, out)):
            return reserve
        return (
            inp * self.settings.input_price + out * self.settings.output_price
        ) / 1e6

    def json(
        self, job: str, prompt: str, validate: Callable[[dict], Any]
    ) -> tuple[Any, dict]:
        request = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only a JSON object matching the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.settings.max_output_tokens,
            "reasoning_effort": self.settings.effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        fingerprint = digest(request)
        last_error = "No complete response"
        for attempt in range(self.settings.api_attempts):
            row = self.store.db.execute(
                "SELECT * FROM calls WHERE job=? AND attempt=?", (job, attempt)
            ).fetchone()
            if row:
                if row["request_hash"] != fingerprint:
                    raise PipelineError(
                        f"Request changed for job {job}; use a new output directory"
                    )
                raw_path = self.store.root / row["raw_path"]
                if raw_path.exists():
                    raw = read_json(raw_path)["response"]
                    self.store.db.execute(
                        "UPDATE calls SET status='complete',cost=? WHERE job=? AND attempt=?",
                        (self._cost(raw, row["cost"]), job, attempt),
                    )
                    self.store.db.commit()
                else:
                    self.store.db.execute(
                        "UPDATE calls SET status='uncertain' WHERE job=? AND attempt=? AND status='requested'",
                        (job, attempt),
                    )
                    self.store.db.commit()
                    last_error = (
                        row["error"] or "Interrupted request; reservation retained"
                    )
                    continue
            else:
                allowance = 2 * len(dumps(request["messages"]).encode("utf-8")) + 4096
                reserve = (
                    allowance * self.settings.input_price
                    + self.settings.max_output_tokens * self.settings.output_price
                ) / 1e6
                if self.store.charge() + reserve > self.budget + 1e-12:
                    raise BudgetReached(
                        f"Next call reserves USD {reserve:.4f}; accounted "
                        f"USD {self.store.charge():.4f} / {self.budget:.2f}. "
                        "Resume with a larger --budget-usd to continue."
                    )
                relative = Path("raw") / (digest(job)[:24] + f"-{attempt}.json")
                self.store.db.execute(
                    "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
                    (
                        job,
                        attempt,
                        fingerprint,
                        "requested",
                        reserve,
                        str(relative),
                        "",
                        now(),
                    ),
                )
                self.store.db.commit()
                try:
                    result = self._client().chat.completions.create(**request)
                    raw = result.model_dump(mode="json")
                    atomic_json(
                        self.store.root / relative,
                        {
                            "job": job,
                            "attempt": attempt,
                            "created": now(),
                            "request": request,
                            "response": raw,
                        },
                    )
                    self.store.db.execute(
                        "UPDATE calls SET status='complete',cost=? WHERE job=? AND attempt=?",
                        (self._cost(raw, reserve), job, attempt),
                    )
                    self.store.db.commit()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    last_error = f"API request failed: {type(exc).__name__}" + (
                        f" (HTTP {status})" if status else ""
                    )
                    self.store.db.execute(
                        "UPDATE calls SET status='uncertain',error=? WHERE job=? AND attempt=?",
                        (last_error, job, attempt),
                    )
                    self.store.db.commit()
                    if isinstance(exc, PipelineError) or status in {
                        400,
                        401,
                        402,
                        403,
                        404,
                        422,
                    }:
                        raise PipelineError(last_error) from exc
                    if attempt + 1 < self.settings.api_attempts:
                        time.sleep(min(2**attempt, 4))
                    continue
            try:
                choice = raw["choices"][0]
                content = choice["message"].get("content")
                if (
                    choice.get("finish_reason") != "stop"
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    raise InvalidCandidate("Empty, refused, or truncated response")
                value = validate(require_object(json.loads(content)))
                return value, {
                    "job": job,
                    "attempt": attempt,
                    "response_id": raw.get("id"),
                    "requested_model": self.settings.model,
                    "returned_model": raw.get("model"),
                    "usage": raw.get("usage"),
                    "request_hash": fingerprint,
                    "effort": self.settings.effort,
                    "pipeline_version": PIPELINE_VERSION,
                }
            except (
                InvalidCandidate,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                self.store.db.execute(
                    "UPDATE calls SET error=? WHERE job=? AND attempt=?",
                    (last_error, job, attempt),
                )
                self.store.db.commit()
        raise InvalidCandidate(f"{job}: {last_error}")


# This trusted harness is mounted into Docker with a small JSON payload.
# Generated code is NEVER executed by the host process.
RUNNER = (
    "ALLOWED_IMPORTS = "
    + repr(sorted(ALLOWED_IMPORTS))
    + "\n"
    + r"""
import builtins
import copy
import json
import math
import platform
import sys

SAFE_NAMES = (
    "abs all any ascii bin bool bytearray bytes callable chr dict divmod enumerate "
    "filter float format frozenset hash hex int isinstance issubclass iter len list "
    "map max min next oct ord pow range repr reversed round set slice sorted str "
    "sum tuple zip Exception ValueError TypeError IndexError KeyError StopIteration "
    "ArithmeticError ZeroDivisionError OverflowError AssertionError"
).split()

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError("Unsupported import")
    if any(x.startswith("__") for x in (fromlist or ())):
        raise ImportError("Unsupported import")
    return builtins.__import__(name, globals, locals, fromlist, 0)

def load(source, name):
    allowed = {key: getattr(builtins, key) for key in SAFE_NAMES}
    allowed["__import__"] = guarded_import
    scope = {"__builtins__": allowed, "__name__": name}
    exec(compile(source, name, "exec"), scope)
    return scope

def json_value(value):
    text = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(text) > 64000:
        raise ValueError("Result too large")
    return json.loads(text)

def equivalent(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    return a == b

def invoke(function, case, immutable=False):
    args, kwargs = copy.deepcopy(case["args"]), copy.deepcopy(case["kwargs"])
    before = json.dumps([args, kwargs], sort_keys=True, allow_nan=False)
    result = function(*args, **kwargs)
    if immutable and json.dumps([args, kwargs], sort_keys=True, allow_nan=False) != before:
        raise ValueError("The solution modified its input arguments")
    return json_value(result)

def verify(payload):
    phase = "checker"
    try:
        reference = load(payload["reference_code"], "reference")["reference"]
        maker = load(payload["input_code"], "input_generator")["make_inputs"]
        cases = payload["cases"]
        for case in cases:
            if not equivalent(invoke(reference, case), case["expected"]):
                return {"passed": False, "phase": phase, "reason": "Reference disagrees with a fixed expected value"}
        extra = maker(payload["random_seed"], payload["fuzz_cases"])
        if not isinstance(extra, list) or len(extra) != payload["fuzz_cases"]:
            return {"passed": False, "phase": phase, "reason": "Input generator returned the wrong number of cases"}
        if json_value(extra) != json_value(maker(payload["random_seed"], payload["fuzz_cases"])):
            return {"passed": False, "phase": phase, "reason": "Input generator is not deterministic"}
        for case in extra:
            if not isinstance(case, dict) or set(case) != {"args", "kwargs"}:
                raise ValueError("Random input must contain args and kwargs")
            if not isinstance(case["args"], list) or not isinstance(case["kwargs"], dict):
                raise ValueError("Invalid random input")
            json_value(case)
        random_cases = [dict(case, expected=invoke(reference, case)) for case in extra]
        phase = "solution"
        candidate = load(payload["code"], "candidate")[payload["entry_point"]]
        checked = 0
        for case in cases + random_cases:
            actual = invoke(candidate, case, immutable=True)
            if not equivalent(actual, case["expected"]):
                return {"passed": False, "phase": phase, "checked": checked,
                        "reason": "Wrong result", "counterexample": json_value(case), "actual": actual}
            checked += 1
        return {"passed": True, "fixed_tests": len(cases), "random_tests": len(extra),
                "checked": checked, "python": platform.python_version()}
    except Exception as exc:
        return {"passed": False, "phase": phase, "reason": type(exc).__name__ + ": " + str(exc)[:500]}

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        payload = json.load(file)
    report = verify(payload)
    print(json.dumps(report, allow_nan=False))
    sys.exit(0 if report["passed"] else 1)
"""
)


class Sandbox:
    def __init__(self, image: str, timeout: int = 30, resolved: bool = False):
        self.timeout = timeout
        if resolved:
            self.image = image
            return
        if not shutil.which("docker"):
            raise PipelineError(
                "Docker is required for verification; install/start it before run"
            )
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise PipelineError(
                f"Docker image {image!r} is unavailable. Start Docker and pull it explicitly first."
            ) from exc
        self.image = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image):
            raise PipelineError("Docker returned an unexpected image identifier")

    def smoke_test(self):
        checker = {
            "reference_code": "def reference(n):\n    return n",
            "input_code": "def make_inputs(seed, count):\n    return [{'args':[i], 'kwargs':{}} for i in range(count)]",
            "cases": [{"args": [0], "kwargs": {}, "expected": 0}],
        }
        report = self.verify(
            {"entry_point": "identity"},
            "def identity(n):\n    return n",
            checker,
            seed=42,
            fuzz_cases=1,
        )
        if not report.get("passed"):
            raise PipelineError(
                "The Docker smoke check failed before any paid generation: "
                + dumps(report)
            )

    def verify(
        self, problem: dict, code: str, checker: dict, seed: int, fuzz_cases: int
    ) -> dict:
        name = "deepseek-distill-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="deepseek-check-") as directory:
            path = Path(directory)
            # The non-root container user needs to traverse the read-only mount.
            path.chmod(0o755)
            payload = {
                **checker,
                "entry_point": problem["entry_point"],
                "code": code,
                "random_seed": seed,
                "fuzz_cases": fuzz_cases,
            }
            atomic_json(path / "payload.json", payload)
            (path / "runner.py").write_text(RUNNER, encoding="utf-8")
            for file in path.iterdir():
                file.chmod(0o444)
            command = [
                "docker",
                "run",
                "--pull=never",
                "--rm",
                "--name",
                name,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=64",
                "--memory=256m",
                "--cpus=1",
                "--user=65534:65534",
                "--log-driver=none",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=32m",
                "--mount",
                f"type=bind,src={path},dst=/work,readonly",
                self.image,
                "python",
                "-I",
                "-B",
                "/work/runner.py",
                "/work/payload.json",
            ]
            try:
                with (
                    tempfile.TemporaryFile() as output,
                    tempfile.TemporaryFile() as errors,
                ):
                    result = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=errors,
                        timeout=self.timeout,
                        check=False,
                    )
                    output.seek(0)
                    data = output.read(65537)
                    if len(data) > 65536:
                        return {
                            "passed": False,
                            "phase": "sandbox",
                            "reason": "Output limit exceeded",
                        }
                    if result.returncode in {125, 126, 127}:
                        raise PipelineError(
                            "Docker could not start the checker; check Docker/image configuration"
                        )
                    try:
                        report = require_object(json.loads(data))
                    except (ValueError, UnicodeError):
                        return {
                            "passed": False,
                            "phase": "sandbox",
                            "reason": "Missing valid checker report",
                        }
                    expected = len(checker["cases"]) + fuzz_cases
                    passed = (
                        result.returncode == 0
                        and report.get("passed") is True
                        and report.get("checked") == expected
                        and report.get("fixed_tests") == len(checker["cases"])
                        and report.get("random_tests") == fuzz_cases
                    )
                    report["passed"] = passed
                    report["image_id"] = self.image
                    return report
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "phase": "sandbox",
                    "reason": "Container time limit exceeded",
                }
            finally:
                # Also removes a timed-out container, which killing the CLI alone may leave running.
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    pass


def load_tokenizer(path: Path):
    if not path.is_dir():
        raise PipelineError(f"Local tokenizer folder not found: {path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PipelineError(
            "Install the project's transformers requirement for Qwen formatting"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.eos_token = EOS
    if tokenizer.convert_tokens_to_ids(EOS) is None:
        raise PipelineError(
            "The local tokenizer does not contain the Qwen ending token"
        )
    if not tokenizer.chat_template:
        raise PipelineError("The local tokenizer has no chat template")
    return tokenizer


def tokenizer_fingerprint(path: Path) -> str:
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        file = path / name
        if file.exists():
            files[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if not files:
        raise PipelineError("No tokenizer files found")
    return digest(files)


def format_for_qwen(instruction: str, response: str, tokenizer) -> dict:
    for text in (instruction, response):
        if any(t in text for t in ("<|im_start|>", EOS, "<|endoftext|>")):
            raise InvalidCandidate("Training content contains chat control tokens")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = response.rstrip() + EOS
    length = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
    return {"prompt": prompt, "completion": completion, "tokens": length}


SCOPE = (
    "Design original self-contained Python 3.12 function tasks using only this "
    "standard-library subset: " + ", ".join(sorted(ALLOWED_IMPORTS)) + ". "
    "Functions must be deterministic and must not modify input arguments. "
    "Inputs/outputs must be JSON-compatible values (string-keyed dictionaries, lists, "
    "strings, finite numbers, booleans, null). Specify behavior for valid inputs; "
    "avoid tasks requiring expected exceptions, files, network, classes, async, "
    "external packages, printing, introspection, or interactive input. "
    "Favor clear basic/intermediate tasks with a smaller challenging tail. "
    "Do not quote, rewrite, or derive tasks from HumanEval, MBPP, or other benchmarks. "
    "Do not put chat markers or Markdown fences in code."
)


def validate_plan(value: dict) -> list[dict]:
    categories = value.get("categories")
    if not isinstance(categories, list) or not 8 <= len(categories) <= 40:
        raise InvalidCandidate("Plan needs 8 to 40 categories")
    result, seen = [], set()
    for category in categories:
        require_object(category)
        name = text_field(category, "name", 100)
        if normalized(name) in seen:
            raise InvalidCandidate("Duplicate category")
        seen.add(normalized(name))
        description = text_field(category, "description", 600)
        weight = category.get("weight", 1)
        if type(weight) is not int or not 1 <= weight <= 5:
            raise InvalidCandidate("Category weight must be 1 to 5")
        result.append(
            {
                "id": digest(name)[:16],
                "name": name,
                "description": description,
                "weight": weight,
            }
        )
    return result


def validate_briefs(value: dict, category: str, count: int) -> list[dict]:
    briefs = value.get("briefs")
    if not isinstance(briefs, list) or not 1 <= len(briefs) <= count:
        raise InvalidCandidate("Unexpected brief count")
    result = []
    for item in briefs:
        require_object(item)
        brief = text_field(item, "brief", 900)
        difficulty = item.get("difficulty")
        if difficulty not in {"basic", "intermediate", "challenging"}:
            raise InvalidCandidate("Unknown difficulty")
        result.append({"category": category, "brief": brief, "difficulty": difficulty})
    if len({normalized(x["brief"]) for x in result}) != len(result):
        raise InvalidCandidate("Repeated briefs in a batch")
    return result


def validate_problem(value: dict) -> dict:
    instruction = text_field(value, "instruction", 7000)
    entry = text_field(value, "entry_point", 100)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", entry) or entry not in instruction:
        raise InvalidCandidate("Instruction must specify the entry point")
    # The trusted harness checks this property, so it must be in the user-visible contract.
    instruction += "\nDo not modify the input arguments."
    return {"instruction": instruction, "entry_point": entry}


def validate_solution(value: dict, entry: str) -> str:
    code = text_field(value, "code", 64000)
    check_code(code, entry)
    return code


def validate_checker(value: dict, settings: Settings) -> dict:
    reference = text_field(value, "reference_code", 40000)
    inputs = text_field(value, "input_code", 20000)
    check_code(reference, "reference")
    check_code(inputs, "make_inputs")
    cases = value.get("cases")
    if not isinstance(cases, list) or not settings.min_tests <= len(cases) <= 64:
        raise InvalidCandidate(f"Need {settings.min_tests} to 64 fixed cases")
    seen = set()
    for case in cases:
        require_object(case)
        if set(case) != {"args", "kwargs", "expected"}:
            raise InvalidCandidate("Each case needs args, kwargs, expected")
        if not isinstance(case["args"], list) or not isinstance(case["kwargs"], dict):
            raise InvalidCandidate("Invalid test arguments")
        if not all(isinstance(k, str) and k.isidentifier() for k in case["kwargs"]):
            raise InvalidCandidate("Keyword argument names must be identifiers")
        if len(dumps(case)) > 16000:
            raise InvalidCandidate("Test case too large")
        key = dumps([case["args"], case["kwargs"]])
        if key in seen:
            raise InvalidCandidate("Repeated fixed inputs")
        seen.add(key)
    return {"reference_code": reference, "input_code": inputs, "cases": cases}


def validate_review(value: dict, nearby: list[dict]) -> dict:
    if type(value.get("approve")) is not bool:
        raise InvalidCandidate("Review needs a boolean approve field")
    reason = text_field(value, "reason", 1500)
    ids = {row["id"] for row in nearby}
    for key in ("duplicate_of", "same_family_as"):
        if value.get(key) is not None and value[key] not in ids:
            raise InvalidCandidate("Review referenced an unknown nearby problem")
    return {
        "approve": value["approve"],
        "reason": reason,
        "duplicate_of": value.get("duplicate_of"),
        "same_family_as": value.get("same_family_as"),
    }


class Exclusions:
    """Local exact/lexical overlap audit. Reserved data is never sent to a teacher."""

    def __init__(self, paths: list[Path]):
        self.manifest, self.texts, self.codes = [], [], set()
        self.exact = set()
        self.postings: dict[str, set[int]] = {}
        for path in paths:
            path = path.resolve()
            count = 0
            with path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        row = require_object(json.loads(line))
                    except (ValueError, TypeError) as exc:
                        raise PipelineError(
                            f"Invalid exclusion JSONL: {path}:{line_number}"
                        ) from exc
                    count += 1
                    texts = [
                        row[k]
                        for k in ("instruction", "question", "prompt")
                        if isinstance(row.get(k), str)
                    ]
                    codes = [
                        row[k]
                        for k in (
                            "code",
                            "response",
                            "completion",
                            "canonical_solution",
                        )
                        if isinstance(row.get(k), str)
                    ]
                    if isinstance(row.get("messages"), list):
                        for message in row["messages"]:
                            if isinstance(message, dict) and isinstance(
                                message.get("content"), str
                            ):
                                (
                                    texts if message.get("role") == "user" else codes
                                ).append(message["content"])
                    if isinstance(row.get("prompt"), str) and isinstance(
                        row.get("canonical_solution"), str
                    ):
                        codes.append(row["prompt"] + row["canonical_solution"])
                    if not texts and not codes:
                        raise PipelineError(
                            f"No recognized text/code fields in exclusion row {path}:{line_number}"
                        )
                    for text in texts:
                        key = normalized(text)
                        self.exact.add(digest(key))
                        index = len(self.texts)
                        self.texts.append(text)
                        for term in words(text):
                            if len(term) > 3:
                                self.postings.setdefault(term, set()).add(index)
                    for code in codes:
                        try:
                            self.codes.add(code_hash(code))
                        except (SyntaxError, ValueError, RecursionError):
                            pass
            self.manifest.append(
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": count,
                }
            )

    def match(self, instruction: str, code: str = "") -> str | None:
        if digest(normalized(instruction)) in self.exact:
            return "exact_reserved_instruction"
        if code and code_hash(code) in self.codes:
            return "exact_reserved_code_ast"
        candidates = set()
        terms = sorted(
            words(instruction), key=lambda w: (len(self.postings.get(w, ())), -len(w))
        )
        for term in [w for w in terms if w in self.postings][:12]:
            candidates.update(self.postings[term])
        for index in candidates:
            if (
                len(words(instruction)) >= 8
                and similarity(instruction, self.texts[index]) >= 0.78
            ):
                return "near_reserved_instruction_lexical"
        return None


class Pipeline:
    def __init__(
        self,
        store: Store,
        teacher: Teacher,
        sandbox: Sandbox,
        tokenizer,
        settings: Settings,
        exclusions: Exclusions,
    ):
        self.store, self.teacher, self.sandbox = store, teacher, sandbox
        self.tokenizer, self.settings, self.exclusions = tokenizer, settings, exclusions

    def plan(self) -> list[dict]:
        path = self.store.root / "plan.json"
        if path.exists():
            return read_json(path)["categories"]
        prompt = (
            SCOPE
            + """
Create a coverage plan of 12 to 24 broad programming categories and subskills.
Choose the topics yourself. Cover diverse operations, representations, and
constraints; include common fundamentals and useful less-common skills.
Weights describe relative coverage, not difficulty.
Return JSON: {"categories":[{"name":"...", "description":"...", "weight":1}]}.
Use integer weights from 1 to 5. Do not generate full questions yet.
"""
        )
        categories, receipt = self.teacher.json("plan", prompt, validate_plan)
        atomic_json(path, {"categories": categories, "provenance": receipt})
        return categories

    def more_seeds(self, categories: list[dict], count: int):
        counts = dict(
            self.store.db.execute(
                "SELECT category,COUNT(*) FROM seeds GROUP BY category"
            )
        )
        category = min(
            categories, key=lambda c: (counts.get(c["id"], 0) / c["weight"], c["id"])
        )
        batch = self.store.db.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        # Freeze prompt context before a paid job so interruptions cannot change it.
        path = self.store.root / "seed_batches" / f"{batch:08d}.json"
        if path.exists():
            spec = read_json(path)
        else:
            recent = [
                r[0]
                for r in self.store.db.execute(
                    "SELECT brief FROM seeds WHERE category=? ORDER BY rowid DESC LIMIT 24",
                    (category["id"],),
                )
            ]
            spec = {"category": category, "count": count, "recent_briefs": recent}
            atomic_json(path, spec)
        category, count = spec["category"], spec["count"]
        prompt = (
            SCOPE
            + "\n"
            + dumps(spec)
            + """
Generate the requested count of distinct problem briefs for this category.
Choose concrete ideas yourself. Each brief should name a core operation,
input/output behavior, and a useful constraint. Avoid existing briefs or
cosmetic variants; sharing a broad algorithm is allowed for distinct tasks.
Favor basic and intermediate tasks. Each brief must be <= 900 characters.
Return JSON: {"briefs":[{"brief":"...", "difficulty":"basic"}]}.
Difficulty must be basic, intermediate, or challenging.
"""
        )
        briefs, receipt = self.teacher.json(
            f"briefs_{batch:08d}",
            prompt,
            lambda v: validate_briefs(v, category["id"], count),
        )
        seen = {
            normalized(r[0]) for r in self.store.db.execute("SELECT brief FROM seeds")
        }
        inserted = 0
        for index, brief in enumerate(briefs):
            key = f"seed_{batch:08d}_{index:03d}"
            state = (
                "duplicate_brief" if normalized(brief["brief"]) in seen else "pending"
            )
            payload = {
                **brief,
                "id": key,
                "category_name": category["name"],
                "provenance": receipt,
            }
            self.store.db.execute(
                "INSERT OR IGNORE INTO seeds VALUES (?,?,?,?,?)",
                (key, category["id"], brief["brief"], dumps(payload), state),
            )
            seen.add(normalized(brief["brief"]))
            inserted += state == "pending"
        self.store.db.commit()
        if inserted == 0:
            raise PipelineError(
                "The brief batch contained only repeats; inspect saved seed batches before resuming"
            )

    def process(self, seed: dict) -> dict:
        key = seed["id"]
        existing = self.store.candidate(key)
        if existing and existing["status"] in {"accepted", "rejected"}:
            return existing
        record = existing or {
            "id": key,
            "seed_id": key,
            "family_id": key,
            "category": seed["category"],
            "category_name": seed["category_name"],
            "difficulty": seed["difficulty"],
            "brief": seed["brief"],
            "status": "processing",
            "created": now(),
            "provenance": {"seed": seed["provenance"]},
            "verification_history": [],
        }
        self.store.save_candidate(record)
        try:
            if "problem" not in record:
                neighbors = self.store.nearest(seed["brief"])
                prompt = (
                    SCOPE
                    + "\n"
                    + dumps({"seed": seed, "nearby_problems": neighbors})
                    + """
Expand the brief into one precise, original problem, meaningfully different
from nearby problems. Ask the user to return code only. Specify a lowercase
function signature, valid input types/bounds, output ordering and edge cases.
All rules that tests may enforce must be stated in the instruction.
Return JSON: {"instruction":"...", "entry_point":"function_name"}.
Do not provide the answer or tests.
"""
                )
                problem, receipt = self.teacher.json(
                    key + "_problem", prompt, validate_problem
                )
                record.update(
                    problem=problem,
                    instruction=problem["instruction"],
                    entry_point=problem["entry_point"],
                )
                record["provenance"]["problem"] = receipt
                self.store.save_candidate(record)
            problem = record["problem"]
            overlap = self.exclusions.match(problem["instruction"])
            duplicate = self.store.exact_duplicate(problem["instruction"])
            if overlap or duplicate:
                raise InvalidCandidate(overlap or f"duplicate_instruction:{duplicate}")
            if "checker" not in record:
                # A fresh call receives ONLY the fixed problem, never a candidate answer.
                prompt = (
                    SCOPE
                    + "\n"
                    + dumps(problem)
                    + f"""
Build an independent checker for the stated problem. You have not been given
the candidate solution. Derive expected values carefully from the specification.
Return JSON with:
1. reference_code: a simple correct implementation named reference, with the
   same arguments as the requested function. A slower straightforward algorithm
   is fine on these small test inputs.
2. cases: {self.settings.min_tests} to 24 DISTINCT cases, each exactly
   {{"args":[], "kwargs":{{}}, "expected":null}}. Cover stated boundary behavior,
   duplicates/order/ties when applicable and plausible wrong implementations.
3. input_code: Python defining make_inputs(seed, count). Use random.Random(seed)
   and return exactly count diverse VALID small input objects, each exactly
   {{"args":[], "kwargs":{{}}}}. Never return expected values or call the solution.
   Respect relationships between arguments and keep the simple reference fast.
No I/O, printing, classes, introspection, or unsupported imports.
"""
                )
                checker, receipt = self.teacher.json(
                    key + "_checker",
                    prompt,
                    lambda v: validate_checker(v, self.settings),
                )
                record["checker"] = checker
                record["provenance"]["checker"] = receipt
                self.store.save_candidate(record)
            checker = record["checker"]
            report = record.get("verification")
            for attempt in range(self.settings.repairs + 1):
                field = f"solution_{attempt}"
                if field not in record:
                    request = {"problem": problem}
                    if attempt:
                        request.update(
                            previous_code=record[f"solution_{attempt - 1}"],
                            failure=record["verification_history"][-1],
                        )
                    prompt = (
                        SCOPE
                        + "\n"
                        + dumps(request)
                        + """
Write a correct complete solution to the fixed problem. If repairing, fix the
code while preserving the specified behavior, function signature and tests.
Return JSON: {"code":"plain Python source"}.
"""
                    )
                    code, receipt = self.teacher.json(
                        key + f"_solution_{attempt}",
                        prompt,
                        lambda v: validate_solution(v, problem["entry_point"]),
                    )
                    record[field] = code
                    record["provenance"][field] = receipt
                    self.store.save_candidate(record)
                code = record[field]
                formatted = format_for_qwen(
                    problem["instruction"], code, self.tokenizer
                )
                if formatted["tokens"] > self.settings.max_example_tokens:
                    raise InvalidCandidate(f"overlength:{formatted['tokens']}")
                overlap = self.exclusions.match(problem["instruction"], code)
                duplicate = self.store.exact_duplicate(problem["instruction"], code)
                if overlap or duplicate:
                    raise InvalidCandidate(overlap or f"duplicate_code:{duplicate}")
                if len(record["verification_history"]) <= attempt:
                    report = self.sandbox.verify(
                        problem,
                        code,
                        checker,
                        int(digest(key)[:8], 16),
                        self.settings.fuzz_cases,
                    )
                    record["verification_history"].append(report)
                    self.store.save_candidate(record)
                else:
                    report = record["verification_history"][attempt]
                if report.get("passed") is True:
                    record.update(
                        response=code, tokens=formatted["tokens"], verification=report
                    )
                    self.store.save_candidate(record)
                    break
                if report.get("phase") != "solution":
                    raise InvalidCandidate(
                        "checker_or_sandbox_failure:" + report.get("reason", "unknown")
                    )
            else:
                raise InvalidCandidate("solution_failed_after_bounded_repairs")
            if "review" not in record:
                neighbors = self.store.nearest(problem["instruction"])
                record["review_context"] = neighbors
                self.store.save_candidate(record)
                prompt = (
                    SCOPE
                    + "\n"
                    + dumps(
                        {
                            "problem": problem,
                            "code": record["response"],
                            "nearby_problems": neighbors,
                        }
                    )
                    + """
Review specification clarity, instruction following, solution correctness and
useful learning value. A test pass is not proof. Reject ambiguous, trivial
rewordings and obvious bugs. Distinct tasks may share an algorithm.
Compare against the nearby problems for semantic duplication. same_family_as
is only for closely related variants, not a whole topic or shared algorithm.
Return JSON: {"approve":true, "reason":"...", "duplicate_of":null,
              "same_family_as":null}.
Non-null IDs must be IDs from nearby_problems.
"""
                )
                review, receipt = self.teacher.json(
                    key + "_review", prompt, lambda v: validate_review(v, neighbors)
                )
                record["review"] = review
                record["provenance"]["review"] = receipt
                self.store.save_candidate(record)
            review = record["review"]
            if not review["approve"] or review["duplicate_of"]:
                raise InvalidCandidate(
                    "quality_or_duplicate_review:" + review["reason"]
                )
            if review["same_family_as"]:
                other = self.store.candidate(review["same_family_as"])
                record["family_id"] = other["family_id"]
            record.update(
                status="accepted",
                completed=now(),
                decontamination="local_exact_and_lexical"
                if self.exclusions.manifest
                else "not_run",
                exclusions=self.exclusions.manifest,
            )
            self.store.save_candidate(record)
            self.store.add_document(record)
        except InvalidCandidate as exc:
            record.update(status="rejected", rejection_reason=str(exc), completed=now())
            self.store.save_candidate(record)
        return record

    def run(self, candidates: int, import_only: bool = False):
        categories = [] if import_only else self.plan()
        self.store.rebuild_documents()
        while True:
            processed = sum(
                self.store.stats()["candidates"].get(s, 0)
                for s in ("accepted", "rejected")
            )
            if processed >= candidates:
                break
            pending = self.store.db.execute(
                "SELECT payload FROM seeds WHERE state='pending' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if not pending:
                if import_only:
                    break
                self.more_seeds(
                    categories,
                    min(self.settings.seed_batch_size, candidates - processed),
                )
                continue
            record = self.process(json.loads(pending[0]))
            stats = self.store.stats()
            print(
                f"{processed + 1}/{candidates} {record['id']} {record['status']} "
                f"accepted={stats['candidates'].get('accepted', 0)} "
                f"accounted_usd={stats['conservative_accounted_usd']:.4f}",
                flush=True,
            )
        atomic_json(self.store.root / "summary.json", self.store.stats())


def import_candidates(store: Store, path: Path) -> int:
    """Reuse collector questions/answers; only missing checks need paid calls."""
    if store.db.execute(
        "SELECT 1 FROM seeds WHERE id NOT LIKE 'import_%' LIMIT 1"
    ).fetchone():
        raise PipelineError("Import candidates into a separate --output directory")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = require_object(json.loads(line))
            if row.get("status") == "rejected":
                continue
            problem = validate_problem(row)
            code = validate_solution(
                {"code": row.get("response", row.get("code"))}, problem["entry_point"]
            )
            key = "import_" + digest([problem, code])[:24]
            if store.candidate(key):
                continue
            origin = {
                "source_file": path.name,
                "source_sha256": source_hash,
                "source_row": line_number,
                "source_id": row.get("id"),
                "generation": row.get("provenance", {}),
            }
            seed = {
                "id": key,
                "category": "imported",
                "category_name": row.get("topic", "Imported"),
                "brief": row.get("brief", problem["instruction"][:900]),
                "difficulty": "intermediate",
                "provenance": origin,
            }
            store.db.execute(
                "INSERT OR IGNORE INTO seeds VALUES (?,?,?,?,?)",
                (key, seed["category"], seed["brief"], dumps(seed), "pending"),
            )
            store.save_candidate(
                {
                    "id": key,
                    "seed_id": key,
                    "family_id": key,
                    "category": seed["category"],
                    "category_name": seed["category_name"],
                    "difficulty": seed["difficulty"],
                    "brief": seed["brief"],
                    "status": "processing",
                    "created": now(),
                    "problem": problem,
                    **problem,
                    "solution_0": code,
                    "provenance": {
                        "seed": origin,
                        "problem": origin,
                        "solution_0": origin,
                    },
                    "verification_history": [],
                }
            )
    count = store.db.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
    if not count:
        raise PipelineError("No valid candidates found to verify")
    return count


def split_for_group(family: str, settings: Settings) -> str:
    fraction = int(digest([settings.split_seed, family])[:16], 16) / 2**64
    if fraction < settings.test_fraction:
        return "test"
    if fraction < settings.test_fraction + settings.validation_fraction:
        return "validation"
    return "train"


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(dumps(row) + "\n")


def export_release(
    store: Store,
    settings: Settings,
    tokenizer,
    exclusions: Exclusions,
    destination: Path,
) -> dict:
    if destination.exists():
        raise PipelineError(f"Refusing to overwrite a release: {destination}")
    accepted = store.rows()
    if not accepted:
        raise PipelineError("There are no accepted examples to export")
    # An exclusion found in one descendant removes the whole problem family.
    excluded_families, excluded = set(), []
    for record in accepted:
        reason = exclusions.match(record["instruction"], record["response"])
        if reason:
            excluded_families.add(record["family_id"])
            excluded.append({"id": record["id"], "reason": reason})
    splits = {name: [] for name in ("train", "validation", "test")}
    source, assignments = [], []
    seen_text, seen_code = set(), set()
    for record in accepted:
        if record["family_id"] in excluded_families:
            excluded.append({"id": record["id"], "reason": "reserved_family"})
            continue
        text_key, code_key = (
            digest(normalized(record["instruction"])),
            code_hash(record["response"]),
        )
        if text_key in seen_text or code_key in seen_code:
            excluded.append({"id": record["id"], "reason": "final_exact_duplicate"})
            continue
        formatted = format_for_qwen(
            record["instruction"], record["response"], tokenizer
        )
        if formatted["tokens"] > settings.max_example_tokens:
            excluded.append({"id": record["id"], "reason": "overlength_at_export"})
            continue
        seen_text.add(text_key)
        seen_code.add(code_key)
        split = split_for_group(record["family_id"], settings)
        source.append({**record, "split": split})
        splits[split].append(
            {"prompt": formatted["prompt"], "completion": formatted["completion"]}
        )
        assignments.append(
            {
                "id": record["id"],
                "family_id": record["family_id"],
                "split": split,
                "tokens": formatted["tokens"],
            }
        )
    if not source:
        raise PipelineError("All accepted examples were removed by the export checks")
    if not splits["train"]:
        raise PipelineError(
            "No training groups in this release; generate more examples"
        )
    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError as exc:
        raise PipelineError(
            "Install the project's datasets requirement for export"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".distill-export-", dir=destination.parent))
    try:
        write_jsonl(staging / "source.jsonl", source)
        write_jsonl(staging / "assignments.jsonl", assignments)
        write_jsonl(staging / "excluded.jsonl", excluded)
        write_jsonl(staging / "rejected.jsonl", store.rows("rejected"))
        features = Features({"prompt": Value("string"), "completion": Value("string")})
        datasets = {}
        hub = staging / "hub"
        (hub / "sft").mkdir(parents=True)
        (hub / "source").mkdir()
        for name, examples in splits.items():
            write_jsonl(staging / f"{name}.jsonl", examples)
            datasets[name] = Dataset.from_dict(
                {key: [row[key] for row in examples] for key in features},
                features=features,
            )
            datasets[name].to_parquet(str(hub / "sft" / f"{name}.parquet"))
            # Heterogeneous test arguments are JSON strings in Arrow to avoid
            # silently coercing types or losing metadata.
            canonical = [
                {
                    "id": r["id"],
                    "family_id": r["family_id"],
                    "category": r["category_name"],
                    "instruction": r["instruction"],
                    "response": r["response"],
                    "metadata_json": dumps(
                        {
                            k: v
                            for k, v in r.items()
                            if k not in {"id", "instruction", "response"}
                        }
                    ),
                }
                for r in source
                if r["split"] == name
            ]
            source_features = Features(
                {
                    key: Value("string")
                    for key in (
                        "id",
                        "family_id",
                        "category",
                        "instruction",
                        "response",
                        "metadata_json",
                    )
                }
            )
            Dataset.from_dict(
                {key: [row[key] for row in canonical] for key in source_features},
                features=source_features,
            ).to_parquet(str(hub / "source" / f"{name}.parquet"))
        DatasetDict(datasets).save_to_disk(str(staging / "sft-splits"))
        lengths = sorted(a["tokens"] for a in assignments)
        manifest = {
            "created": now(),
            "pipeline_version": PIPELINE_VERSION,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": asdict(settings),
            "run": store.stats(),
            "exported": len(source),
            "splits": {k: len(v) for k, v in splits.items()},
            "families": len({r["family_id"] for r in source}),
            "categories": dict(Counter(r["category_name"] for r in source)),
            "export_exclusions": excluded,
            "reserved_corpora": exclusions.manifest,
            "decontamination": "local_exact_and_lexical"
            if exclusions.manifest
            else "not_run",
            "tokens": {
                "total": sum(lengths),
                "min": lengths[0],
                "median": lengths[len(lengths) // 2],
                "max": lengths[-1],
            },
            "limitations": [
                "Teacher-generated expected values, reference implementations and reviews may share errors.",
                "Passing the generated checker is not a correctness proof or a human quality audit.",
                "Duplicate retrieval is lexical/AST plus teacher review of retrieved neighbors, not exhaustive semantic search.",
                "An exclusion corpus is supplied locally; absent corpora have not been audited.",
                "Exact/lexical benchmark checks cannot certify absence of semantic contamination.",
                "Conservative usage accounting is not a provider-enforced billing cap.",
                "Test is a held-out split; use validation, not test, during development.",
            ],
        }
        card = """---
task_categories:
- text-generation
tags:
- python
- synthetic-data
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: sft/train.parquet
  - split: validation
    path: sft/validation.parquet
  - split: test
    path: sft/test.parquet
- config_name: source
  data_files:
  - split: train
    path: source/train.parquet
  - split: validation
    path: source/validation.parquet
  - split: test
    path: source/test.parquet
---
# Synthetic Python function dataset

Generated with DeepSeek and selected using generated fixed tests, deterministic
randomized reference comparisons, scope checks, and a teacher quality review.
See manifest.json for actual settings, counts, costs, audit coverage and limits.

The source configuration preserves instruction/response and metadata_json,
including generated checkers and provenance. The sft configuration contains
Qwen prompt/completion text with completion-only targets and an ending marker.
No human review or exhaustive semantic contamination audit is implied.
No dataset license is assigned automatically.
"""
        (hub / "README.md").write_text(card, encoding="utf-8")
        (staging / "README.md").write_text(card, encoding="utf-8")
        for name in ("openai", "transformers", "datasets", "huggingface-hub"):
            try:
                manifest.setdefault("packages", {})[name] = importlib.metadata.version(
                    name
                )
            except importlib.metadata.PackageNotFoundError:
                pass
        atomic_json(hub / "manifest.json", manifest)
        manifest["files"] = {
            str(file.relative_to(staging)): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in sorted(staging.rglob("*"))
            if file.is_file()
        }
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_release(path: Path) -> dict:
    manifest = require_object(read_json(path / "manifest.json"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PipelineError("Release has no integrity manifest")
    root = path.resolve()
    actual_files = {
        str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
    }
    if actual_files != set(files) | {"manifest.json"}:
        raise PipelineError("Release contains missing or unmanifested files")
    for name, expected in files.items():
        file = (path / name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise PipelineError(f"Missing or invalid release file: {name}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise PipelineError(f"Release file was changed: {name}")
    return manifest


def upload_release(path: Path, repo_id: str, public: bool = False):
    verify_release(path)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise PipelineError(
            "Use a Hugging Face dataset ID such as username/python-distill-v1"
        )
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError as exc:
        raise PipelineError(
            "Install the project's huggingface-hub requirement"
        ) from exc
    api = HfApi()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        if info.private != (not public):
            raise PipelineError(
                "Existing repository visibility differs; choose a new repo or matching --public setting"
            )
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id, repo_type="dataset", private=not public, exist_ok=False
        )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(path / "hub"),
        commit_message="Upload verified synthetic Python dataset release",
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    run = commands.add_parser("run", help="Generate candidates; calls the paid API")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument(
        "--import-candidates",
        type=Path,
        help="Verify an existing collector JSONL instead of regenerating its questions/answers",
    )
    run.add_argument(
        "--candidates",
        type=positive_int,
        default=30000,
        help="Total attempted candidates including previous completed rows, not accepted-row target",
    )
    run.add_argument(
        "--budget-usd",
        type=positive_float,
        required=True,
        help="Total conservative run-directory budget, including previous calls",
    )
    run.add_argument("--model", choices=("deepseek-flash",), default="deepseek-flash")
    run.add_argument("--effort", choices=("low", "high", "max"), default="high")
    run.add_argument("--max-output-tokens", type=positive_int, default=8192)
    run.add_argument("--max-example-tokens", type=positive_int, default=2048)
    run.add_argument("--api-attempts", type=positive_int, default=3)
    run.add_argument("--repairs", type=nonnegative_int, default=1)
    run.add_argument("--seed-batch-size", type=positive_int, default=12)
    run.add_argument("--min-tests", type=positive_int, default=8)
    run.add_argument("--fuzz-cases", type=nonnegative_int, default=32)
    run.add_argument("--sandbox-timeout", type=positive_int, default=30)
    run.add_argument("--image", default="python:3.12-slim")
    run.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    run.add_argument(
        "--input-price",
        type=positive_float,
        default=0.30,
        help="Conservative USD per 1M input tokens; verify current provider rates",
    )
    run.add_argument(
        "--output-price",
        type=positive_float,
        default=1.20,
        help="Conservative USD per 1M billed output tokens, including reasoning",
    )
    run.add_argument("--split-seed", type=int, default=42)
    run.add_argument("--validation-fraction", type=float, default=0.05)
    run.add_argument("--test-fraction", type=float, default=0.05)
    run.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Local reserved prompts/solutions; repeat for multiple corpora; never sent to teacher",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without API calls or file writes",
    )
    check = commands.add_parser(
        "check", help="Check local dependencies, tokenizer and cached Docker image"
    )
    check.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    check.add_argument("--image", default="python:3.12-slim")
    status = commands.add_parser(
        "status", help="Read saved progress without calling APIs"
    )
    status.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    export = commands.add_parser(
        "export", help="Create a frozen local source/SFT release; no API calls"
    )
    export.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    export.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    export.add_argument("--release", type=Path)
    export.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    upload = commands.add_parser(
        "upload", help="Upload an existing frozen release to Hugging Face"
    )
    upload.add_argument("--release", type=Path, required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument(
        "--public",
        action="store_true",
        help="Use a public repository; default is private",
    )
    return parser


def ensure_configuration(store: Store, settings: Settings, exclusions: Exclusions):
    path = store.root / "config.json"
    config = {"settings": asdict(settings), "exclusions": exclusions.manifest}
    if path.exists():
        if read_json(path) != config:
            raise PipelineError(
                "Saved settings/tokenizer/image/exclusion corpora differ. "
                "Use the original settings to resume or choose a new --output."
            )
    else:
        atomic_json(path, config)


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "upload":
            upload_release(args.release.resolve(), args.repo_id, args.public)
            return 0
        if args.command == "check":
            try:
                import openai
            except ImportError as exc:
                raise PipelineError(
                    "Install requirements-distill.txt before using the API"
                ) from exc
            tokenizer = load_tokenizer(args.tokenizer.resolve())
            sandbox = Sandbox(args.image)
            sandbox.smoke_test()
            print(
                dumps(
                    {
                        "openai": openai.__version__,
                        "image_id": sandbox.image,
                        "tokenizer_hash": tokenizer_fingerprint(
                            args.tokenizer.resolve()
                        ),
                        "example_tokens": format_for_qwen(
                            "Return one.", "def one():\n    return 1", tokenizer
                        )["tokens"],
                    }
                )
            )
            return 0
        if args.command == "run":
            if not all(
                math.isfinite(v) and 0 <= v < 1
                for v in (args.validation_fraction, args.test_fraction)
            ):
                raise PipelineError(
                    "Split fractions must be finite and between 0 and 1"
                )
            if args.validation_fraction + args.test_fraction >= 1:
                raise PipelineError(
                    "Validation and test fractions must leave a training split"
                )
            if (
                not 3 <= args.min_tests <= 64
                or args.fuzz_cases > 256
                or args.seed_batch_size > 32
            ):
                raise PipelineError(
                    "Use 3..64 fixed tests, <=256 random tests and <=32 briefs per batch"
                )
            if args.dry_run:
                print(
                    dumps(
                        {
                            "paid_calls": False,
                            "output": str(args.output.resolve()),
                            "model": args.model,
                            "candidates": args.candidates,
                            "budget_usd": args.budget_usd,
                            "stages": [
                                "plan",
                                "briefs",
                                "problem",
                                "checker",
                                "solution/repair",
                                "review",
                            ],
                            "next": "Remove --dry-run after local setup; exports/uploads are separate commands",
                        }
                    )
                )
                return 0
            tokenizer = load_tokenizer(args.tokenizer.resolve())
            sandbox = Sandbox(args.image, args.sandbox_timeout)
            sandbox.smoke_test()
            exclusions = Exclusions(args.exclude_jsonl)
            settings = Settings(
                **{
                    field: getattr(args, field)
                    for field in Settings.__dataclass_fields__
                    if hasattr(args, field)
                },
                image_id=sandbox.image,
                tokenizer_hash=tokenizer_fingerprint(args.tokenizer.resolve()),
            )
            with writer_lock(args.output.resolve()):
                store = Store(args.output.resolve())
                try:
                    ensure_configuration(store, settings, exclusions)
                    target = args.candidates
                    if args.import_candidates:
                        target = min(
                            target, import_candidates(store, args.import_candidates)
                        )
                    teacher = Teacher(store, settings, args.budget_usd)
                    teacher._client()  # Dependency/key failures happen before reserving a request.
                    if not exclusions.manifest:
                        print(
                            "No reserved corpus supplied: benchmark decontamination is unverified.",
                            flush=True,
                        )
                    Pipeline(
                        store, teacher, sandbox, tokenizer, settings, exclusions
                    ).run(target, import_only=bool(args.import_candidates))
                    print(dumps(store.stats()))
                finally:
                    store.close()
            return 0
        if not (args.output / "state.sqlite").is_file():
            raise PipelineError(f"No saved run at {args.output}")
        if args.command == "status":
            store = Store(args.output.resolve())
            try:
                print(json.dumps(store.stats(), indent=2))
            finally:
                store.close()
            return 0
        if args.command == "export":
            settings = Settings(**read_json(args.output / "config.json")["settings"])
            tokenizer = load_tokenizer(args.tokenizer.resolve())
            if (
                tokenizer_fingerprint(args.tokenizer.resolve())
                != settings.tokenizer_hash
            ):
                raise PipelineError(
                    "Export tokenizer differs from the generation tokenizer"
                )
            exclusions = Exclusions(args.exclude_jsonl)
            if not exclusions.manifest:
                print(
                    "No export exclusion corpus: release will explicitly record decontamination=not_run."
                )
            destination = args.release or (
                args.output
                / "releases"
                / (
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    + "-"
                    + uuid.uuid4().hex[:8]
                )
            )
            with writer_lock(args.output.resolve()):
                store = Store(args.output.resolve())
                try:
                    manifest = export_release(
                        store, settings, tokenizer, exclusions, destination.resolve()
                    )
                    print(
                        dumps(
                            {
                                "release": str(destination.resolve()),
                                "splits": manifest["splits"],
                                "tokens": manifest["tokens"],
                            }
                        )
                    )
                finally:
                    store.close()
            return 0
        raise PipelineError("Unknown command")
    except BudgetReached as exc:
        print(f"Paused at budget: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Stopped. Saved responses and completed candidates can be resumed.",
            file=sys.stderr,
        )
        return 130
    except (PipelineError, InvalidCandidate, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
