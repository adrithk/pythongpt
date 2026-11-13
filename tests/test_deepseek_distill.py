"""Offline checks for the small collector; no paid calls or code execution."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import test_deepseek_pipeline as fixtures

import deepseek_distill as d


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = {"output": self.root, "effort": "high", "budget": 10, "spent": 0}

    def tearDown(self):
        self.temporary.cleanup()

    def test_request_cached_without_charging_again(self):
        client = fixtures.FakeClient([{"value": "hello"}])
        value, _ = d.request_json(
            client, self.state, "one", "Return JSON", {"value": str}
        )
        before = self.state["spent"]
        again, _ = d.request_json(
            client, self.state, "one", "Return JSON", {"value": str}
        )
        self.assertEqual(value, again)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.state["spent"], before)
        self.assertAlmostEqual(before, 0.00009)
        with self.assertRaises(RuntimeError):
            d.request_json(client, self.state, "one", "Changed prompt", {"value": str})

    def test_budget_stops_before_paid_call(self):
        self.state["budget"] = 0.0000001
        client = fixtures.FakeClient([{"value": "hello"}])
        with self.assertRaises(d.BudgetLimit):
            d.request_json(client, self.state, "one", "Return JSON", {"value": str})
        self.assertEqual(client.requests, [])

    def test_bad_json_and_missing_fields_retry_boundedly(self):
        client = fixtures.FakeClient(
            [
                fixtures.raw("  "),
                fixtures.raw('{"other":1}'),
                {"value": "recovered"},
            ]
        )
        with self.assertRaises(RuntimeError):
            d.request_json(client, self.state, "one", "Return JSON", {"value": str})
        self.assertEqual(len(client.requests), 2)
        value, _ = d.request_json(
            client, self.state, "one", "Return JSON", {"value": str}
        )
        self.assertEqual(value["value"], "recovered")
        self.assertEqual(len(client.requests), 3)

    def test_interrupted_request_keeps_reservation(self):
        client = fixtures.FakeClient([KeyboardInterrupt(), {"value": "ok"}])
        with self.assertRaises(KeyboardInterrupt):
            d.request_json(client, self.state, "one", "Return JSON", {"value": str})
        reserve = self.state["spent"]
        value, _ = d.request_json(
            client, self.state, "one", "Return JSON", {"value": str}
        )
        self.assertEqual(value["value"], "ok")
        self.assertGreater(self.state["spent"], reserve)

    def test_dictionary_syntax_and_exact_duplicate_checks(self):
        answer = {**fixtures.PROBLEM, "code": fixtures.GOOD}
        seen = set()
        record = d.make_record(
            answer, {"topic": "lists", "brief": "sort integers"}, 0, {}, seen
        )
        self.assertEqual(record["instruction"], fixtures.PROBLEM["instruction"])
        self.assertEqual(record["response"], fixtures.GOOD)
        self.assertEqual(record["status"], "unverified")
        self.assertNotIn("completion", record)
        with self.assertRaises(ValueError):
            d.make_record(answer, {}, 1, {}, seen)
        with self.assertRaises((ValueError, SyntaxError)):
            d.make_record({**answer, "code": "broken python !!!"}, {}, 1, {}, set())

    def test_full_collection_resume_and_saved_jsonl(self):
        client = fixtures.FakeClient(
            [
                {"topics": [f"topic-{i}" for i in range(8)]},
                {"briefs": [f"Different useful task {i}" for i in range(d.BATCH_SIZE)]},
                {**fixtures.PROBLEM, "code": fixtures.GOOD},
            ]
        )
        args = SimpleNamespace(
            output=self.root, effort="high", candidates=1, budget_usd=10
        )
        with contextlib.redirect_stdout(io.StringIO()):
            d.collect(client, args)
            d.collect(client, args)
        rows = [
            json.loads(line)
            for line in (self.root / "candidates.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "unverified")
        self.assertEqual(len(client.requests), 3)
        args.effort = "low"
        with self.assertRaises(RuntimeError):
            d.collect(client, args)

    def test_dry_run_does_not_create_files_or_require_dependencies(self):
        destination = self.root / "absent"
        with contextlib.redirect_stdout(io.StringIO()):
            result = d.main(
                [
                    "--candidates",
                    "1",
                    "--budget-usd",
                    "1",
                    "--output",
                    str(destination),
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
