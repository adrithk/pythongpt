"""Offline tests. All code fixtures here are authored, trusted test inputs.

No model/API calls, package installs, downloads, or real Docker runs are made.
The embedded harness is exercised locally ONLY with these trusted fixtures.
"""

import contextlib
import copy
import io
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import deepseek_pipeline as d

GOOD = "def sorted_copy(nums):\n    return sorted(nums)"
BAD = "def sorted_copy(nums):\n    return sorted(set(nums))"
REFERENCE = "def reference(nums):\n    return sorted(nums)"
INPUTS = (
    "import random\n"
    "def make_inputs(seed, count):\n"
    "    rng = random.Random(seed)\n"
    "    return [{'args': [[rng.randint(-8, 8) for _ in range(rng.randint(0, 8))]], "
    "'kwargs': {}} for _ in range(count)]"
)
LISTS = [[], [1], [2, 1], [2, 2, 1], [-2, 0, 2], [3, 2, 1], [0], [4, -1, 4, 0]]
CHECKER = {
    "reference_code": REFERENCE,
    "input_code": INPUTS,
    "cases": [{"args": [x], "kwargs": {}, "expected": sorted(x)} for x in LISTS],
}
PROBLEM = {
    "instruction": "Write sorted_copy(nums) returning the integers in ascending order, "
    "preserving duplicates, including an empty list. Return Python code only.",
    "entry_point": "sorted_copy",
}


class ToyTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return (
            "<|im_start|>user\n"
            + messages[0]["content"]
            + d.EOS
            + "\n<|im_start|>assistant\n"
        )

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


class FakeClient:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if not self.replies:
            raise AssertionError("Unexpected additional API request")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        raw = (
            reply
            if isinstance(reply, dict) and "choices" in reply
            else {
                "id": "fake-response",
                "model": "fake-deepseek",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(reply)}}
                ],
            }
        )
        return SimpleNamespace(model_dump=lambda **_: copy.deepcopy(raw))


def raw(content, finish="stop", usage=True):
    return {
        "id": "raw-test",
        "model": "fake",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50} if usage else None,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def harness_verify(problem, code, checker, seed=42, fuzz_cases=8):
    namespace = {"__name__": "trusted_test_harness"}
    # Only this repository's authored fixtures are executed in this offline test.
    exec(compile(d.RUNNER, "<trusted-harness>", "exec"), namespace)  # noqa: S102
    return namespace["verify"](
        {
            **copy.deepcopy(checker),
            "entry_point": problem["entry_point"],
            "code": code,
            "random_seed": seed,
            "fuzz_cases": fuzz_cases,
        }
    )


class TrustedFixtureSandbox:
    def __init__(self):
        self.calls = 0

    def verify(self, problem, code, checker, seed, fuzz_cases):
        self.calls += 1
        return harness_verify(problem, code, checker, seed, fuzz_cases)


def accepted_record(key, instruction, code, family=None):
    return {
        "id": key,
        "seed_id": key,
        "family_id": family or key,
        "category": "lists",
        "category_name": "Lists",
        "instruction": instruction,
        "response": code,
        "status": "accepted",
        "verification": {"passed": True},
    }


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = d.Store(self.root)
        self.settings = d.Settings(api_attempts=2, fuzz_cases=8)

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def teacher(self, replies, budget=10):
        client = FakeClient(replies)
        return d.Teacher(self.store, self.settings, budget, client), client


class StoreTest(StoreFixture):
    def test_import_reuses_paid_question_and_answer(self):
        path = self.root / "candidates.jsonl"
        row = {**PROBLEM, "response": GOOD, "status": "unverified", "topic": "Lists"}
        path.write_text(d.dumps(row) + "\n", encoding="utf-8")
        count = d.import_candidates(self.store, path)
        teacher, client = self.teacher(
            [
                CHECKER,
                {
                    "approve": True,
                    "reason": "Correct and distinct",
                    "duplicate_of": None,
                    "same_family_as": None,
                },
            ]
        )
        pipeline = d.Pipeline(
            self.store,
            teacher,
            TrustedFixtureSandbox(),
            ToyTokenizer(),
            self.settings,
            d.Exclusions([]),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.run(count, import_only=True)
        self.assertEqual(self.store.stats()["candidates"]["accepted"], 1)
        self.assertEqual(len(client.requests), 2)
        self.assertNotIn(GOOD, client.requests[0]["messages"][1]["content"])
        self.assertEqual(d.import_candidates(self.store, path), 1)
        self.assertFalse((self.root / "plan.json").exists())

    def test_paid_response_cache_and_usage(self):
        teacher, client = self.teacher([{"value": 1}])
        first, receipt = teacher.json("one", "Return JSON", lambda v: v)
        second, _ = teacher.json("one", "Return JSON", lambda v: v)
        self.assertEqual(first, second)
        self.assertEqual(len(client.requests), 1)
        self.assertAlmostEqual(self.store.charge(), 0.00009)
        self.assertTrue(receipt["request_hash"])
        request = client.requests[0]
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertTrue(list((self.root / "raw").glob("*.json")))
        with self.assertRaises(d.PipelineError):
            teacher.json("one", "Changed prompt", lambda v: v)
        self.assertEqual(len(client.requests), 1)

    def test_blank_truncated_and_schema_failures_are_bounded_and_saved(self):
        for label, reply in [
            ("blank", raw("   ")),
            ("truncated", raw('{"a":', "length")),
            ("schema", raw('{"wrong":true}')),
        ]:
            teacher, client = self.teacher([reply, {"value": 4}])

            def validate(v):
                if "value" not in v:
                    raise d.InvalidCandidate("missing value")
                return v

            result, _ = teacher.json(label, "Return JSON", validate)
            self.assertEqual(result["value"], 4)
            self.assertEqual(len(client.requests), 2)
        self.assertEqual(self.store.stats()["api_attempts"], 6)

    def test_budget_prevents_request(self):
        teacher, client = self.teacher([{"value": 1}], budget=0.000001)
        with self.assertRaises(d.BudgetReached):
            teacher.json("over-budget", "Return JSON", lambda v: v)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.store.charge(), 0)

    def test_missing_usage_is_charged_at_reserved_ceiling(self):
        teacher, _ = self.teacher([raw('{"value":1}', usage=False)])
        teacher.json("unknown-usage", "Return JSON", lambda v: v)
        self.assertGreater(self.store.charge(), 0.009)

    def test_interrupted_call_retains_reservation_and_does_not_reissue_same_attempt(
        self,
    ):
        teacher, client = self.teacher([KeyboardInterrupt(), {"value": 2}])
        with self.assertRaises(KeyboardInterrupt):
            teacher.json("interrupt", "Return JSON", lambda v: v)
        reserved = self.store.charge()
        result, _ = teacher.json("interrupt", "Return JSON", lambda v: v)
        self.assertEqual(result["value"], 2)
        self.assertEqual(len(client.requests), 2)
        self.assertGreater(self.store.charge(), reserved)
        self.assertEqual(self.store.stats()["uncertain_calls"], 1)

    def test_auth_error_is_not_retried_or_leaked(self):
        class AuthError(Exception):
            status_code = 401

        teacher, client = self.teacher([AuthError("secret-key-must-not-be-printed")])
        with self.assertRaises(d.PipelineError) as error:
            teacher.json("auth", "Return JSON", lambda v: v)
        self.assertNotIn("secret-key", str(error.exception))
        self.assertEqual(len(client.requests), 1)

    def test_lexical_search_exact_ast_and_index_recovery(self):
        record = accepted_record("one", PROBLEM["instruction"], GOOD)
        self.store.save_candidate(record)
        self.store.rebuild_documents()
        self.assertEqual(self.store.exact_duplicate(PROBLEM["instruction"]), "one")
        self.assertEqual(
            self.store.exact_duplicate("different instruction", GOOD + "\n# comment"),
            "one",
        )
        self.assertEqual(
            self.store.nearest("ascending integers sorted duplicates")[0]["id"], "one"
        )
        self.store.rebuild_documents()
        self.assertEqual(
            self.store.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1
        )

    def test_pipeline_full_flow_repairs_fixed_checker_and_resumes(self):
        seed = {
            "id": "seed-test",
            "category": "lists",
            "category_name": "Lists",
            "brief": "Return sorted integers preserving repeats",
            "difficulty": "basic",
            "provenance": {},
        }
        self.store.db.execute(
            "INSERT INTO seeds VALUES (?,?,?,?,?)",
            (seed["id"], seed["category"], seed["brief"], d.dumps(seed), "pending"),
        )
        self.store.db.commit()
        replies = [
            PROBLEM,
            CHECKER,
            {"code": BAD},
            {"code": GOOD},
            {
                "approve": True,
                "reason": "Correct and clear",
                "duplicate_of": None,
                "same_family_as": None,
            },
        ]
        teacher, client = self.teacher(replies)
        sandbox = TrustedFixtureSandbox()
        pipeline = d.Pipeline(
            self.store,
            teacher,
            sandbox,
            ToyTokenizer(),
            self.settings,
            d.Exclusions([]),
        )
        record = pipeline.process(seed)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["response"], GOOD)
        self.assertEqual(len(record["verification_history"]), 2)
        self.assertEqual(record["checker"], CHECKER)
        self.assertNotIn(GOOD, client.requests[1]["messages"][1]["content"])
        self.assertEqual(sandbox.calls, 2)
        before = len(client.requests)
        self.assertEqual(pipeline.process(seed)["status"], "accepted")
        self.assertEqual(len(client.requests), before)
        self.assertEqual(sandbox.calls, 2)

    def test_scope_screening(self):
        for code in [
            "import os\ndef bad(x):\n    return x",
            "def bad(x):\n    return eval(x)",
            "def bad(x):\n    return x.__class__",
            "def bad(x):\n    raise SystemExit(0)",
        ]:
            with self.assertRaises(d.InvalidCandidate):
                d.check_code(code, "bad")
        d.check_code(GOOD, "sorted_copy")

    def test_exclusion_sources_and_near_overlap(self):
        path = self.root / "reserved.jsonl"
        path.write_text(
            d.dumps({"instruction": PROBLEM["instruction"], "code": GOOD}) + "\n"
        )
        exclusions = d.Exclusions([path])
        self.assertTrue(exclusions.match(PROBLEM["instruction"]))
        self.assertTrue(exclusions.match("another question", GOOD))
        self.assertIsNone(
            exclusions.match(
                "Compute a graph adjacency matrix for a directed edge list"
            )
        )
        self.assertEqual(exclusions.manifest[0]["rows"], 1)

    def test_configuration_changes_refused(self):
        d.ensure_configuration(self.store, self.settings, d.Exclusions([]))
        d.ensure_configuration(self.store, self.settings, d.Exclusions([]))
        with self.assertRaises(d.PipelineError):
            d.ensure_configuration(
                self.store, replace(self.settings, effort="low"), d.Exclusions([])
            )

    def test_controller_chooses_topics_and_does_not_regenerate_completed_target(self):
        plan = {
            "categories": [
                {
                    "name": f"Category {i}",
                    "description": f"Distinct programming skill {i}",
                    "weight": 1,
                }
                for i in range(8)
            ]
        }
        replies = [
            plan,
            {
                "briefs": [
                    {
                        "brief": "Sort integers preserving duplicates",
                        "difficulty": "basic",
                    }
                ]
            },
            PROBLEM,
            CHECKER,
            {"code": GOOD},
            {
                "approve": True,
                "reason": "Clear",
                "duplicate_of": None,
                "same_family_as": None,
            },
        ]
        teacher, client = self.teacher(replies)
        pipeline = d.Pipeline(
            self.store,
            teacher,
            TrustedFixtureSandbox(),
            ToyTokenizer(),
            self.settings,
            d.Exclusions([]),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.run(1)
            pipeline.run(1)
        self.assertEqual(self.store.stats()["candidates"]["accepted"], 1)
        self.assertEqual(len(client.requests), 6)
        self.assertTrue((self.root / "plan.json").exists())
        self.assertTrue((self.root / "summary.json").exists())

    def test_candidate_resumes_after_interrupt_with_frozen_problem_and_checker(self):
        seed = {
            "id": "resume-seed",
            "category": "lists",
            "category_name": "Lists",
            "brief": "Order integer lists",
            "difficulty": "basic",
            "provenance": {},
        }
        teacher, client = self.teacher(
            [
                PROBLEM,
                CHECKER,
                KeyboardInterrupt(),
                {"code": GOOD},
                {
                    "approve": True,
                    "reason": "Clear",
                    "duplicate_of": None,
                    "same_family_as": None,
                },
            ]
        )
        pipeline = d.Pipeline(
            self.store,
            teacher,
            TrustedFixtureSandbox(),
            ToyTokenizer(),
            self.settings,
            d.Exclusions([]),
        )
        with self.assertRaises(KeyboardInterrupt):
            pipeline.process(seed)
        partial = self.store.candidate(seed["id"])
        self.assertEqual(partial["status"], "processing")
        problem = copy.deepcopy(partial["problem"])
        checker = copy.deepcopy(partial["checker"])
        result = pipeline.process(seed)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["problem"], problem)
        self.assertEqual(result["checker"], checker)
        self.assertEqual(len(client.requests), 5)


class HarnessTest(unittest.TestCase):
    def test_good_bad_and_checker_errors(self):
        good = harness_verify(PROBLEM, GOOD, CHECKER)
        self.assertTrue(good["passed"])
        self.assertEqual(good["checked"], 16)
        self.assertFalse(harness_verify(PROBLEM, BAD, CHECKER)["passed"])
        bad_checker = copy.deepcopy(CHECKER)
        bad_checker["cases"][0]["expected"] = [99]
        result = harness_verify(PROBLEM, GOOD, bad_checker)
        self.assertFalse(result["passed"])
        self.assertEqual(result["phase"], "checker")

    def test_input_mutation_rejected(self):
        code = "def sorted_copy(nums):\n    nums.sort()\n    return nums"
        result = harness_verify(PROBLEM, code, CHECKER)
        self.assertFalse(result["passed"])
        self.assertIn("modified", result["reason"])

    def test_boolean_is_not_integer_zero(self):
        problem = {"entry_point": "zero"}
        checker = {
            "reference_code": "def reference():\n    return 0",
            "input_code": "def make_inputs(seed, count):\n    return []",
            "cases": [{"args": [], "kwargs": {}, "expected": 0}],
        }
        self.assertFalse(
            harness_verify(
                problem, "def zero():\n    return False", checker, fuzz_cases=0
            )["passed"]
        )

    def test_docker_flags_report_integrity_and_timeout_cleanup(self):
        sandbox = d.Sandbox("sha256:" + "a" * 64, resolved=True)
        requests = []

        def run(command, **kwargs):
            requests.append(command)
            if command[1] == "run":
                payload_path = (
                    next(v for v in command if v.startswith("type=bind,src="))
                    .split("src=")[1]
                    .split(",dst=")[0]
                )
                payload = json.loads((Path(payload_path) / "payload.json").read_text())
                report = {
                    "passed": True,
                    "checked": len(payload["cases"]) + payload["fuzz_cases"],
                    "fixed_tests": len(payload["cases"]),
                    "random_tests": payload["fuzz_cases"],
                }
                kwargs["stdout"].write(json.dumps(report).encode())
            return SimpleNamespace(returncode=0)

        with patch.object(d.subprocess, "run", side_effect=run):
            report = sandbox.verify(PROBLEM, GOOD, CHECKER, 42, 8)
        self.assertTrue(report["passed"])
        for flag in (
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--pull=never",
            "--user=65534:65534",
            "--memory=256m",
        ):
            self.assertIn(flag, requests[0])
        self.assertEqual(requests[-1][1:3], ["rm", "-f"])

        def no_report(command, **kwargs):
            return SimpleNamespace(returncode=0)

        with patch.object(d.subprocess, "run", side_effect=no_report):
            self.assertFalse(sandbox.verify(PROBLEM, GOOD, CHECKER, 42, 8)["passed"])

        def timeout(command, **kwargs):
            if command[1] == "run":
                raise subprocess.TimeoutExpired(command, 1)
            return SimpleNamespace(returncode=0)

        with patch.object(d.subprocess, "run", side_effect=timeout) as mock_run:
            self.assertFalse(sandbox.verify(PROBLEM, GOOD, CHECKER, 42, 8)["passed"])
            self.assertEqual(mock_run.call_args_list[-1].args[0][1:3], ["rm", "-f"])


class ExportTest(StoreFixture):
    """Also exercise the real installed Datasets/Arrow export libraries."""

    def test_group_splits_export_integrity_and_no_target_leakage(self):
        record = accepted_record(
            "one",
            "Write a function returning the integer one.",
            "def one():\n    return 1",
            "same-family",
        )
        record["checker"] = CHECKER
        record["teacher_reasoning"] = "PRIVATE_TEST_REASONING"
        record2 = accepted_record(
            "two",
            "Write a function returning the integer two.",
            "def two():\n    return 2",
            "same-family",
        )
        self.store.save_candidate(record)
        self.store.save_candidate(record2)
        settings = replace(self.settings, validation_fraction=0, test_fraction=0)
        release = self.root / "release"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            manifest = d.export_release(
                self.store, settings, ToyTokenizer(), d.Exclusions([]), release
            )
        self.assertEqual(manifest["exported"], 2)
        self.assertEqual(manifest["splits"]["train"], 2)
        self.assertTrue((release / "hub" / "sft" / "train.parquet").is_file())
        self.assertTrue((release / "sft-splits" / "dataset_dict.json").is_file())
        from datasets import Dataset, load_from_disk

        restored = load_from_disk(str(release / "sft-splits"))
        self.assertEqual(set(restored), {"train", "validation", "test"})
        self.assertEqual(len(restored["train"]), 2)
        source_view = Dataset.from_parquet(
            str(release / "hub" / "source" / "train.parquet"),
            cache_dir=str(self.root / "cache"),
        )
        self.assertEqual(len(source_view), 2)
        self.assertIn("checker", json.loads(source_view[0]["metadata_json"]))
        training = (release / "train.jsonl").read_text()
        self.assertNotIn("PRIVATE_TEST_REASONING", training)
        self.assertNotIn("reference_code", training)
        for line in training.splitlines():
            row = json.loads(line)
            self.assertEqual(set(row), {"prompt", "completion"})
            self.assertTrue(row["completion"].endswith(d.EOS))
        self.assertEqual(d.verify_release(release)["exported"], 2)
        with self.assertRaises(d.PipelineError):
            d.export_release(
                self.store, settings, ToyTokenizer(), d.Exclusions([]), release
            )
        (release / "train.jsonl").write_text("tampered")
        with self.assertRaises(d.PipelineError):
            d.verify_release(release)

    def test_reserved_match_removes_entire_family(self):
        for record in [
            accepted_record(
                "one", "Return one", "def one():\n    return 1", "reserved-family"
            ),
            accepted_record(
                "two", "Return two", "def two():\n    return 2", "reserved-family"
            ),
            accepted_record(
                "three", "Return three", "def three():\n    return 3", "other-family"
            ),
        ]:
            self.store.save_candidate(record)
        path = self.root / "reserved.jsonl"
        path.write_text(d.dumps({"instruction": "Return one"}) + "\n")
        settings = replace(self.settings, validation_fraction=0, test_fraction=0)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            manifest = d.export_release(
                self.store,
                settings,
                ToyTokenizer(),
                d.Exclusions([path]),
                self.root / "clean",
            )
        self.assertEqual(manifest["exported"], 1)

    def test_private_upload_checks_files_and_visibility_before_network_mutation(self):
        self.store.save_candidate(
            accepted_record("one", "Return one", "def one():\n    return 1")
        )
        settings = replace(self.settings, validation_fraction=0, test_fraction=0)
        release = self.root / "upload"
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            d.export_release(
                self.store, settings, ToyTokenizer(), d.Exclusions([]), release
            )
        from unittest.mock import MagicMock

        api = MagicMock()
        api.repo_info.return_value = SimpleNamespace(private=True)
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            d.upload_release(release, "unit/python-data")
        api.upload_folder.assert_called_once()
        self.assertEqual(
            api.upload_folder.call_args.kwargs["folder_path"], str(release / "hub")
        )
        api.reset_mock()
        api.repo_info.return_value = SimpleNamespace(private=False)
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            self.assertRaises(d.PipelineError),
        ):
            d.upload_release(release, "unit/python-data")
        api.upload_folder.assert_not_called()
        (release / "hub" / "unexpected.env").write_text("do not upload")
        with self.assertRaises(d.PipelineError):
            d.verify_release(release)


class CLITest(unittest.TestCase):
    def test_dry_run_has_no_files_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "does-not-exist"
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(d.Teacher, "_client") as client,
            ):
                result = d.main(
                    [
                        "run",
                        "--candidates",
                        "3",
                        "--budget-usd",
                        "1",
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            client.assert_not_called()

    def test_invalid_settings_fail_before_dependencies(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                d.main(
                    [
                        "run",
                        "--budget-usd",
                        "1",
                        "--validation-fraction",
                        "nan",
                        "--dry-run",
                    ]
                ),
                1,
            )
        self.assertEqual(
            d.split_for_group("a", d.Settings()), d.split_for_group("a", d.Settings())
        )


@unittest.skipUnless(
    (d.DEFAULT_TOKENIZER / "tokenizer_config.json").is_file(), "No local Qwen tokenizer"
)
class QwenTokenizerTest(unittest.TestCase):
    def test_actual_local_tokenizer_format_and_length(self):
        tokenizer = d.load_tokenizer(d.DEFAULT_TOKENIZER)
        result = d.format_for_qwen(
            "Return the integer one.", "def one():\n    return 1", tokenizer
        )
        self.assertTrue(result["prompt"].endswith("<|im_start|>assistant\n"))
        self.assertEqual(result["completion"], "def one():\n    return 1" + d.EOS)
        self.assertEqual(
            result["tokens"],
            len(
                tokenizer(
                    result["prompt"] + result["completion"], add_special_tokens=False
                )["input_ids"]
            ),
        )
        with self.assertRaises(d.InvalidCandidate):
            d.format_for_qwen(
                "Return one.", "def one():\n    return 1" + d.EOS, tokenizer
            )


if __name__ == "__main__":
    unittest.main()
