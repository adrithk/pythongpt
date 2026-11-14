"""Offline checks for the three curation steps.

Fixture rows copy the column layout of the real sources. The local Python
interpreter stands in for Docker and runs only the fixture code written here.
No downloads and no containers.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import curate_build as b
import curate_collect as c
import curate_verify as v
from distill_common import read_jsonl, write_jsonl

TOKENIZER_PRESENT = (b.TOKENIZER / "tokenizer.json").exists()


def local_command(docker, image, program=v.RUNNER, limit=v.TIME_LIMIT):
    return [sys.executable, "-I", "-c", program]


def quietly(function, *arguments):
    with contextlib.redirect_stdout(io.StringIO()) as text:
        result = function(*arguments)
    return result, text.getvalue()


def row(source, number, instruction=None, code=None, tests=("assert True",)):
    name = f"task_{source}_{number}"
    return {
        "id": f"{source}:{number}",
        "source": source,
        "instruction": instruction
        or f"Write {name} adding {number} zeta{number} eta{number} theta{number}.",
        "response": code or f"def {name}(x):\n    return x + {number}",
        "tests": list(tests),
    }


class CollectTest(unittest.TestCase):
    def test_clean_code_keeps_definitions_only(self):
        code = (
            "import math\nfrom collections import Counter\n\nLIMIT = 3\n\n"
            "@cache\ndef f(x):\n    # note\n    return math.sqrt(x)\n\n"
            "print(f(4))\nif __name__ == '__main__':\n    f(1)\n"
        )
        self.assertEqual(
            c.clean_code(code),
            "import math\nfrom collections import Counter\nLIMIT = 3\n\n\n"
            "@cache\ndef f(x):\n    # note\n    return math.sqrt(x)",
        )
        two = c.clean_code("def a():\n    return 1\nclass B:\n    pass\n")
        self.assertEqual(two, "def a():\n    return 1\n\n\nclass B:\n    pass")

    def test_clean_code_rejects_out_of_scope(self):
        for code in [
            "import numpy as np\ndef f():\n    pass",
            "import os\ndef f():\n    pass",
            "from os.path import join\ndef f():\n    pass",
            "from . import helper\ndef f():\n    pass",
            "def f():\n    import requests\n    return 1",
            "def f():\n    return input()",
            "def f(path):\n    return open(path).read()",
            "x = 1\nprint(x)",
            "def f(:",
        ]:
            with self.assertRaises(ValueError, msg=code):
                c.clean_code(code)

    def test_educational_rows(self):
        raw = {
            "seq_id": 7,
            "instruction": "Write a function to check if a given string is a palindrome.",
            "output": "Here is the code to solve this problem:\n```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```",
            "code": "def is_palindrome(s):\n    return s == s[::-1]",
            "entry_point": "is_palindrome",
            "testcase": [
                'assert is_palindrome("racecar") == True',
                "",
                'assert is_palindrome("ab") == False',
            ],
        }
        made = c.make_row("educational", *c.from_educational(raw))
        self.assertEqual(made["id"], "educational:7")
        self.assertEqual(made["response"], raw["code"])
        self.assertEqual(len(made["tests"]), 2)
        raw["testcase"] = [
            "node = [1, 2]",
            "# setup",
            "assert is_palindrome(node) == False",
        ]
        made = c.make_row("educational", *c.from_educational(raw))
        self.assertEqual(
            made["tests"],
            ["node = [1, 2]\n# setup\nassert is_palindrome(node) == False"],
        )
        raw["testcase"] = ["for x in (1, 2:", "assert True"]  # Not valid Python.
        with self.assertRaises(ValueError):
            c.make_row("educational", *c.from_educational(raw))

    def opencode_row(self, **changes):
        raw = {
            "id": "4d61",
            "input": "You are given a list of tasks. Return how many do not overlap.",
            "output": "```python\ndef count_tasks(tasks):\n    return len(tasks)\n\n# Example usage:\nprint(count_tasks([]))\n```",
            "domain": "generic",
            "generation_algorithm": "self-instruct",
            "llm_judgement": json.dumps(
                {
                    "requirement_conformance": {"score": 5, "justification": "ok"},
                    "logical_correctness": {"score": 5, "justification": "ok"},
                }
            ),
            "unit_tests": '["\\nassert count_tasks([]) == 0\\n", "\\nassert count_tasks([(1, 2)]) == 1\\n"]',
            "tests_execution_status": '["pass", "pass"]',
            "average_test_score": "1",
        }
        raw.update(changes)
        return raw

    def test_opencode_rows(self):
        made = c.make_row("opencode", *c.from_opencode(self.opencode_row()))
        self.assertEqual(
            made["response"], "def count_tasks(tasks):\n    return len(tasks)"
        )
        self.assertEqual(
            made["tests"],
            ["assert count_tasks([]) == 0", "assert count_tasks([(1, 2)]) == 1"],
        )
        single_quoted = "{'a': {'score': 5, 'justification': \"it's fine\"}}"
        c.from_opencode(
            self.opencode_row(llm_judgement=single_quoted, average_test_score=1.0)
        )
        for changes in [
            {"average_test_score": "0.9"},
            {"llm_judgement": json.dumps({"a": {"score": 4}})},
            {"llm_judgement": "not a structure"},
            {"output": "No code here."},
        ]:
            with self.assertRaises(ValueError, msg=changes):
                c.from_opencode(self.opencode_row(**changes))

    def test_selfoss_rows(self):
        raw = {
            "id": 165808,
            "instruction": "Write a Python function named `get_value`.",
            "response": "To handle errors we check bounds.\n\nHere is the function:\n\n```python\ndef get_value(m, i):\n    return m[i] if i < len(m) else None\n```\n\nIt returns None when out of range.\n\n```python\nassert get_value([1], 0) == 1\nassert get_value([1], 5) is None\n```",
            "seed": "def other(): pass",
            "concepts": [],
            "prompt": "Provide the best response...",
        }
        made = c.make_row("selfoss", *c.from_selfoss(raw))
        self.assertTrue(made["response"].startswith("def get_value"))
        self.assertEqual(len(made["tests"]), 2)
        raw["response"] = "```csharp\nint x;\n```"
        with self.assertRaises(ValueError):
            c.from_selfoss(raw)

    def test_convert_counts_reasons_and_stops(self):
        rows = [self.opencode_row(id=str(n)) for n in range(5)]
        rows.insert(1, self.opencode_row(average_test_score="0.5"))
        rows.insert(2, {"id": "broken"})
        kept, reasons = c.convert("opencode", rows, take=3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(reasons["a recorded test failed upstream"], 1)
        self.assertEqual(reasons["unexpected row shape"], 1)


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patch = mock.patch.object(v, "docker_command", local_command)
        patch.start()
        self.addCleanup(patch.stop)
        self.args = SimpleNamespace(
            source="educational", output=self.root, workers=2, docker="true", image="x"
        )
        self.rows = [
            row("educational", 0, tests=["assert task_educational_0(1) == 1"]),
            row("educational", 1, tests=["assert task_educational_1(1) == 99"]),
            row("educational", 2, tests=[]),
            row(
                "educational",
                3,
                code="import not_installed_anywhere\ndef f():\n    pass",
                tests=[],
            ),
        ]

    def test_batch_runner_judges_each_row(self):
        command = local_command(None, None, v.BATCH_RUNNER)
        stuck = row(
            "educational",
            9,
            code="def f():\n    while True:\n        pass",
            tests=["f()"],
        )
        self.assertEqual(
            v.run_batch(command, [*self.rows, stuck], limit=2),
            [True, False, True, False, False],
        )
        self.assertIsNone(
            v.run_batch([sys.executable, "-I", "-c", "raise SystemExit(1)"], self.rows)
        )

    def test_verify_writes_passes_and_resumes(self):
        (self.root / "candidates").mkdir()
        write_jsonl(self.root / "candidates" / "educational.jsonl", self.rows)
        _, text = quietly(v.verify, self.args)
        kept = read_jsonl(self.root / "verified" / "educational.jsonl")
        self.assertEqual([r["id"] for r in kept], ["educational:0", "educational:2"])
        self.assertIn("4 of 4 rows still to test", text)
        _, text = quietly(v.verify, self.args)
        self.assertIn("0 of 4 rows still to test", text)

    def test_broken_container_falls_back_to_single_rows(self):
        (self.root / "candidates").mkdir()
        write_jsonl(self.root / "candidates" / "educational.jsonl", self.rows)
        with mock.patch.object(v, "run_batch", lambda command, rows: None):
            quietly(v.verify, self.args)
        kept = read_jsonl(self.root / "verified" / "educational.jsonl")
        self.assertEqual([r["id"] for r in kept], ["educational:0", "educational:2"])


@unittest.skipUnless(TOKENIZER_PRESENT, "local Qwen tokenizer not present")
class BuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "data" / "curated"
        (self.out / "verified").mkdir(parents=True)
        benchmark = [
            (
                "bench_fn",
                b.words(
                    "Return the largest gap between neighbouring sorted numbers in the list"
                ),
            )
        ]
        for patch in [
            mock.patch.object(b, "ROOT", self.root),
            mock.patch.object(
                b, "QUOTAS", {"deepseek": 5, "opencode": 40, "educational": 20}
            ),
            mock.patch.object(b, "DEV_ROWS", 6),
            mock.patch.object(b, "NAME_CAP", 3),
            mock.patch.object(b, "load_benchmarks", lambda: benchmark),
        ]:
            patch.start()
            self.addCleanup(patch.stop)
        self.args = SimpleNamespace(
            name="t", seed=0, output=self.out, skip_benchmark_audit=False
        )

    def test_build_filters_then_fills_quotas(self):
        deepseek = [row("deepseek", n) for n in range(8)]
        opencode = [row("opencode", n) for n in range(30)]
        opencode += [
            row("opencode", 100, code="def bench_fn(x):\n    return x"),
            row(
                "opencode",
                101,
                instruction="Return the largest gap between neighbouring sorted numbers in a list",
            ),
            row(
                "opencode",
                102,
                code="def long_one(x):\n    return " + " + ".join(["x"] * 1200),
            ),
        ]
        opencode += [  # Same function name five times with unrelated wording: capped at three.
            row(
                "opencode",
                200 + n,
                instruction=f"Solve puzzle kappa{n} lambda{n} mu{n} nu{n} xi{n}.",
                code=f"def solve(x):\n    return x * {n + 2}",
            )
            for n in range(5)
        ]
        educational = [row("educational", n, tests=[]) for n in range(10)]
        educational += [
            dict(
                row("educational", 50), instruction=deepseek[0]["instruction"]
            ),  # Exact repeat of text.
            dict(
                row("educational", 51), response=deepseek[1]["response"]
            ),  # Exact repeat of code.
            row(
                "educational",
                52,
                instruction=opencode[0]["instruction"] + " Please.",
                code="def task_opencode_0(x):\n    return x - 1",
            ),
        ]
        for name, rows in [
            ("deepseek", deepseek),
            ("opencode", opencode),
            ("educational", educational),
        ]:
            write_jsonl(self.out / "verified" / f"{name}.jsonl", rows)
        quietly(b.build, self.args)

        report = json.loads((self.out / "t-report.json").read_text())
        removed = report["removed"]
        self.assertEqual(removed["educational: exact repeat"], 2)
        self.assertEqual(removed["educational: near repeat"], 1)
        self.assertEqual(removed["opencode: benchmark function name"], 1)
        self.assertEqual(removed["opencode: benchmark wording"], 1)
        self.assertEqual(removed["opencode: too long"], 1)
        self.assertEqual(removed["opencode: function name used too often"], 2)
        self.assertEqual(report["benchmark_check"], "run against 1 tasks")

        train, dev = (
            read_jsonl(self.out / "t-train.jsonl"),
            read_jsonl(self.out / "t-dev.jsonl"),
        )
        chosen = count_by_source(train + dev)
        self.assertEqual(chosen, {"deepseek": 5, "opencode": 33, "educational": 10})
        self.assertTrue(dev and all(r["tests"] for r in dev))
        self.assertFalse({r["id"] for r in dev} & {r["id"] for r in train})

        from datasets import load_from_disk

        splits = load_from_disk(str(self.root / "data" / "t-splits"))
        self.assertEqual(
            (len(splits["train"]), len(splits["test"])), (len(train), len(dev))
        )
        self.assertEqual(splits["train"].column_names, ["prompt", "completion"])
        first = splits["train"][0]
        self.assertTrue(first["prompt"].endswith("<|im_start|>assistant\n"))
        self.assertIn(train[0]["instruction"], first["prompt"])
        self.assertEqual(first["completion"], train[0]["response"] + "<|im_end|>")

        with self.assertRaises(RuntimeError):  # Never overwrites an existing set.
            quietly(b.build, self.args)

    def test_skipping_the_benchmark_check_is_recorded(self):
        write_jsonl(
            self.out / "verified" / "opencode.jsonl",
            [row("opencode", n) for n in range(12)],
        )
        self.args.skip_benchmark_audit = True
        quietly(b.build, self.args)
        self.assertEqual(
            json.loads((self.out / "t-report.json").read_text())["benchmark_check"],
            "NOT RUN",
        )


def count_by_source(rows):
    counts = {}
    for item in rows:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    return counts


if __name__ == "__main__":
    unittest.main()
