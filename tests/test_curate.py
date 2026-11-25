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

    def test_kodcode_rows(self):
        raw = {
            "style": "complete",
            "question_id": "Leetcode_1",
            "question": 'def double(x: int) -> int:\n    """ Return twice x.\n    >>> double(2)\n    4\n    """\n',
            "solution": "def double(x):\n    # Twice x.\n    return 2 * x\n\nprint(double(2))",
            "test": "from solution import double\nimport math\n\n"
            "def test_small():\n    assert double(2) == 4\n\n"
            "def test_zero():\n    assert double(0) == 0\n",
        }
        made = c.make_row("kodcode", *c.from_kodcode(raw))
        self.assertTrue(made["instruction"].startswith("Implement the following"))
        self.assertEqual(
            made["response"], "def double(x):\n    # Twice x.\n    return 2 * x"
        )
        self.assertEqual(len(made["tests"]), 2)
        space = {}
        exec(made["response"], space)  # noqa: S102 (trusted fixture)
        for test in made["tests"]:
            self.assertIn("import math", test)
            exec(test, dict(space))  # noqa: S102 (trusted fixture: defines, calls a test)
        for changes in [
            {"style": "instruct"},
            {"test": "import pytest\n\ndef test_a():\n    assert True\n"},
            {"test": "def test_a(tmp_path):\n    assert tmp_path\n"},
            {"test": "from solution import double\n"},
        ]:
            with self.assertRaises(ValueError, msg=changes):
                c.from_kodcode({**raw, **changes})

    def test_kodcode_questions_lose_pasted_tests(self):
        stub = 'from typing import List\n\ndef double(x: int) -> int:\n    """ Twice x. """\n    pass\n'
        tails = [
            "\n\nfrom solution import double\n\ndef test_a():\n    assert double(2) == 4\n",
            "\n# Unit tests\ndef test_a():\n    assert double(2) == 4\n",
            "\nfrom typing import List\nimport pytest\n\ndef test_a():\n    assert double(2) == 4\n",
            "\nclass TestDouble:\n    def test_a(self):\n        assert double(2) == 4\n",
        ]
        for tail in tails:
            self.assertEqual(c.strip_kodcode_tests(stub + tail), stub, msg=tail)
        self.assertEqual(c.strip_kodcode_tests(stub), stub)  # Nothing to cut.
        self.assertEqual(c.strip_kodcode_tests("import pytest\n\n" + stub), stub)
        made = c.from_kodcode(
            {
                "style": "complete",
                "question_id": "Leetcode_2",
                "question": stub + tails[0],
                "solution": "def double(x):\n    return 2 * x",
                "test": "def test_a():\n    assert double(2) == 4\n",
            }
        )
        self.assertNotIn("assert", made[1])
        for leftover in [
            '    """\n    Unit Tests:\n    from solution import double\n    """\n',
            "\n# Tests\nassert double(2) == 4\n",
        ]:
            with self.assertRaises(ValueError, msg=leftover):
                c.strip_kodcode_tests(stub.replace("    pass\n", leftover))

    def test_opencode_edge_case_minimum(self):
        def judged(edge, logic):
            return json.dumps(
                {
                    "requirement_conformance": {"score": 5},
                    "logical_correctness": {"score": logic},
                    "edge_case_consideration": {"score": edge},
                }
            )

        edge4 = self.opencode_row(llm_judgement=judged(4, 5))
        logic4 = self.opencode_row(llm_judgement=judged(5, 4))
        with self.assertRaises(ValueError):  # Default: edge case must be 5.
            c.from_opencode(edge4)
        with mock.patch.object(c, "min_edge_score", 4):
            c.from_opencode(edge4)  # Loosened: edge case 4 is accepted...
            with self.assertRaises(ValueError):  # ...but other scores stay at 5.
                c.from_opencode(logic4)
            with self.assertRaises(ValueError):
                c.from_opencode(self.opencode_row(llm_judgement=judged(3, 5)))

    def test_opencode_reads_local_copy(self):
        import datasets.config
        import pyarrow as pa
        import pyarrow.parquet as pq

        import load_opencode

        rows = [self.opencode_row(id=str(n)) for n in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "raw"
            (raw / "data").mkdir(parents=True)
            for part in range(2):  # Two files, as if 2 of the 50 were downloaded.
                name = f"train-{part:05d}-of-00050.parquet"
                pq.write_table(
                    pa.Table.from_pylist(rows[part * 10 : part * 10 + 10]),
                    raw / "data" / name,
                )
            with (
                mock.patch.object(load_opencode, "RAW", raw),
                mock.patch.object(datasets.config, "HF_DATASETS_CACHE", folder),
            ):
                in_order, printed = quietly(
                    lambda: [r["id"] for r in c.stream("opencode", 0, False)]
                )
                mixed = [
                    r["id"] for r in quietly(list, c.stream("opencode", 0, True))[0]
                ]
                again = [
                    r["id"] for r in quietly(list, c.stream("opencode", 0, True))[0]
                ]
        self.assertEqual(in_order, [str(n) for n in range(20)])
        self.assertIn("Reading 2 of 50", printed)
        self.assertIn("Not all files", printed)
        self.assertNotEqual(mixed, in_order)
        self.assertEqual(sorted(mixed, key=int), in_order)
        self.assertEqual(mixed, again)  # Same seed, same order.

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

    def test_build_continuing_from_a_previous_set(self):
        # First set: 20 opencode rows.
        first = [row("opencode", n) for n in range(20)]
        write_jsonl(self.out / "verified" / "opencode.jsonl", first)
        quietly(b.build, self.args)
        old_train = read_jsonl(self.out / "t-train.jsonl")
        old_dev = read_jsonl(self.out / "t-dev.jsonl")

        # Second set: the same 20 rows, 10 new rows, and one new row that
        # repeats an old row's code.
        new = [row("opencode", n) for n in range(100, 110)]
        repeat = dict(row("opencode", 200), response=first[3]["response"])
        write_jsonl(self.out / "verified" / "opencode.jsonl", first + new + [repeat])
        args = SimpleNamespace(
            **vars(self.args), previous=str(self.out / "t"), replay=0.2
        )
        args.name = "t2"
        quietly(b.build, args)

        train = read_jsonl(self.out / "t2-train.jsonl")
        dev = read_jsonl(self.out / "t2-dev.jsonl")
        new_ids = {r["id"] for r in new}
        old_train_ids = {r["id"] for r in old_train}
        self.assertEqual([r["id"] for r in dev], [r["id"] for r in old_dev])
        self.assertEqual({r["id"] for r in train} & new_ids, new_ids)
        self.assertNotIn(repeat["id"], {r["id"] for r in train})
        replayed = [r for r in train if r["id"] in old_train_ids]
        self.assertEqual(len(replayed), 2)  # 20% of the 10 new rows.
        self.assertFalse({r["id"] for r in dev} & {r["id"] for r in train})

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
