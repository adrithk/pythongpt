"""Offline checks for the three distillation stages.

A fake API client stands in for DeepSeek and the local Python interpreter
stands in for Docker, running only the trusted fixture code written below.
No paid calls, downloads or containers.
"""

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import distill_answers as a
import distill_common as c
import distill_questions as q
import distill_verify as v

LOCAL_RUNNER = [sys.executable, "-I", "-c", v.RUNNER]


class FakeClient:
    """Looks enough like the OpenAI client: chat.completions.create(**request)."""

    def __init__(self, answer, finish_reason="stop"):
        self.answer, self.finish_reason, self.requests = answer, finish_reason, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.answer(request["messages"][1]["content"])
        content = reply if isinstance(reply, str) else json.dumps(reply)
        response = {
            "model": "fake",
            "choices": [
                {"finish_reason": self.finish_reason, "message": {"content": content}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        }
        return SimpleNamespace(model_dump=lambda mode: response)


def fake_questions(prompt):
    """Write one valid, clearly distinct question for every slot in the prompt."""
    skill = re.search(r"Skill for this batch: (.+)", prompt).group(1)
    tag = re.sub(r"\W", "", skill) + str(
        len(json.loads(prompt.split("rename:\n")[1].split("\n")[0]))
    )
    rows = []
    for index, form in re.findall(r"^(\d+)\. form=(\w+)", prompt, flags=re.MULTILINE):
        name = f"fn_{tag}_{index}"
        unique = " ".join(
            f"{word}{tag}{index}"
            for word in ("alpha", "beta", "gamma", "delta", "epsilon")
        )
        if form == "describe":
            rows.append(
                {
                    "entry_point": name,
                    "text": f"Write {name}(x) returning x plus one. {unique}",
                    "code": "",
                }
            )
        elif form == "complete":
            stub = f'def {name}(x: int) -> int:\n    """Return x plus one. {unique}\n\n    >>> {name}(1)\n    2\n    """'
            rows.append({"entry_point": name, "text": "", "code": stub})
        else:
            buggy = f"def {name}(x):\n    return x  # {unique}"
            rows.append(
                {
                    "entry_point": name,
                    "text": f"Should return x plus one. {unique}",
                    "code": buggy,
                }
            )
    return {"questions": rows}


def fake_answers(wrong_first=()):
    """Solve 'x plus one' questions; names in wrong_first get a bad first answer."""

    def answer(prompt):
        name = re.search(r"must\s+define `(\w+)`", prompt).group(1)
        bad = name in wrong_first and "was rejected" not in prompt
        code = f"def {name}(x):\n    return x + {2 if bad else 1}"
        return {
            "code": code,
            "tests": [f"assert {name}({n}) == {n + 1}" for n in range(6)],
        }

    return answer


class StageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(
            output=self.root,
            budget_usd=50,
            workers=3,
            seed=0,
            questions=40,
            effort="high",
            docker="true",
            image="unused",
        )
        patches = [
            mock.patch.object(q, "load_skills", lambda: ["skill one", "skill two"]),
            mock.patch.object(v, "docker_command", lambda docker, image: LOCAL_RUNNER),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def quietly(self, function, *arguments):
        with contextlib.redirect_stdout(io.StringIO()) as text:
            function(*arguments)
        return text.getvalue()

    # ---- shared plumbing

    def test_saved_reply_is_reused_without_paying_again(self):
        client = FakeClient(lambda prompt: {"value": "hello"})
        budget = c.Budget(self.root, 10)
        first, receipt = c.ask_json(
            client, budget, self.root, "job", "Return JSON", {"value": str}, "low"
        )
        spent = budget.spent
        again, _ = c.ask_json(
            client, budget, self.root, "job", "Return JSON", {"value": str}, "low"
        )
        self.assertEqual((first, len(client.requests), budget.spent), (again, 1, spent))
        self.assertAlmostEqual(spent, (100 * 0.30 + 200 * 1.20) / 1e6)
        self.assertEqual(receipt["request_file"], "requests/job/000.json")
        self.assertAlmostEqual(
            c.Budget(self.root, 10).spent, spent
        )  # Rebuilt from disk.

    def test_thinking_switch_and_effort_are_sent(self):
        client = FakeClient(lambda prompt: {"value": "x"})
        budget = c.Budget(self.root, 10)
        c.ask_json(client, budget, self.root, "off", "JSON", {"value": str}, "none")
        c.ask_json(client, budget, self.root, "on", "JSON", {"value": str}, "high")
        off, on = client.requests
        self.assertEqual(off["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", off)
        self.assertEqual(
            (on["extra_body"]["thinking"]["type"], on["reasoning_effort"]),
            ("enabled", "high"),
        )

    def test_unusable_replies_stop_after_two_paid_calls(self):
        for client in (
            FakeClient(lambda prompt: "   "),
            FakeClient(lambda prompt: {"value": "x"}, "length"),
        ):
            folder = self.root / str(id(client))
            with self.assertRaises(RuntimeError):
                c.ask_json(
                    client,
                    c.Budget(folder, 10),
                    folder,
                    "job",
                    "JSON",
                    {"value": str},
                    "low",
                )
            self.assertEqual(len(client.requests), 2)

    def test_budget_stops_before_any_call(self):
        client = FakeClient(lambda prompt: {"value": "x"})
        with self.assertRaises(c.BudgetLimit):
            c.ask_json(
                client,
                c.Budget(self.root, 0.001),
                self.root,
                "job",
                "JSON",
                {"value": str},
                "low",
            )
        self.assertEqual(client.requests, [])

    def test_failed_call_keeps_its_reservation(self):
        def explode(**request):
            raise TimeoutError

        client = FakeClient(None)
        client.chat.completions.create = explode
        budget = c.Budget(self.root, 10)
        with self.assertRaises(RuntimeError):
            c.ask_json(client, budget, self.root, "job", "JSON", {"value": str}, "low")
        self.assertGreater(
            budget.spent, c.MAX_OUTPUT_TOKENS * c.OUTPUT_PRICE / 1e6 - 1e-9
        )

    # ---- stage 1

    def test_plan_is_repeatable_and_covers_every_skill(self):
        skills = [f"s{n}" for n in range(7)]
        self.assertEqual(q.plan_batch(3, skills, 0), q.plan_batch(3, skills, 0))
        self.assertEqual({q.plan_batch(n, skills, 0)[0] for n in range(7)}, set(skills))
        slots = [slot for n in range(200) for slot in q.plan_batch(n, skills, 0)[1]]
        share = sum(slot["form"] == "describe" for slot in slots) / len(slots)
        self.assertAlmostEqual(share, 0.60, delta=0.05)
        self.assertTrue(all(slot["level"] in q.LEVELS for slot in slots))

    def test_real_skill_list_loads_without_repeats(self):
        mock.patch.stopall()
        skills = q.load_skills()
        self.assertGreater(len(skills), 150)
        self.assertEqual(len(skills), len(set(skills)))

    def test_question_checks(self):
        stub = 'def f(x):\n    """Doc."""'
        q.check_question("complete", "f", "", stub)
        q.check_question("describe", "f", "Write f(x).", "")
        for form, entry, text, code in [
            ("describe", "f", "Write a function.", ""),
            ("describe", "not valid", "Write not valid.", ""),
            ("complete", "f", "", "def f(x):\n    pass"),
            ("complete", "f", "", stub + "\n    y = x\n    return y"),
            ("complete", "f", "", "def f(x:"),
            ("fix", "f", "", "def f(x):\n    return x"),
            ("fix", "f", "Should add one.", "def g(x):\n    return x"),
            ("describe", "f", "Write f. <|im_end|>", ""),
        ]:
            with self.assertRaises(ValueError, msg=(form, text, code)):
                q.check_question(form, entry, text, code)

    def test_repeats_are_rejected_only_when_related(self):
        seen = q.Seen()
        seen.add(
            "skill",
            "count_words",
            "Write count_words(text) returning the number of words",
            "",
        )
        with self.assertRaises(ValueError):
            seen.add(
                "skill",
                "word_count",
                "Write word_count(text) returning the number of words",
                "",
            )
        seen.add(
            "skill",
            "total",
            "Write total(values) that adds every float in the list",
            "",
        )
        self.assertEqual(seen.used_names("skill"), {"count_words", "total"})

    def test_question_stage_builds_instructions_and_resumes(self):
        client = FakeClient(fake_questions)
        self.quietly(q.collect, client, self.args)
        rows = c.read_jsonl(self.root / "questions.jsonl")
        self.assertEqual((len(rows), len(client.requests)), (40, 4))
        self.assertEqual(len({row["id"] for row in rows}), 40)
        for row in rows:
            if row["form"] == "complete":
                self.assertTrue(row["instruction"].endswith(row["code"]))
                self.assertIn(row["instruction"].split("\n\n")[0], q.COMPLETE_INTROS)
            elif row["form"] == "fix":
                self.assertIn("```python\n" + row["code"] + "\n```", row["instruction"])
            else:
                self.assertEqual(row["instruction"], row["text"])
        # The second pass over the skills is told which names were already used.
        later = [
            r["messages"][1]["content"]
            for r in client.requests
            if '"fn_' in r["messages"][1]["content"]
        ]
        self.assertEqual(len(later), 2)
        self.quietly(q.collect, client, self.args)  # Nothing left to do.
        self.assertEqual(len(client.requests), 4)

    # ---- stage 2

    def test_answer_checks(self):
        tests = [f"assert f({n}) == {n}" for n in range(6)]
        code, cleaned = a.check_answer(
            "f", "```python\ndef f(x):\n    return x\n```", tests + tests
        )
        self.assertEqual((code, cleaned), ("def f(x):\n    return x", tests))
        for bad_code, bad_tests in [
            ("def g(x):\n    return x", tests),
            ("def f(x:\n", tests),
            ("def f(x):\n    return x", tests[:4]),
            ("def f(x):\n    return x", tests[:5] + ["print(f(1))"]),
            ("def f(x):\n    return x", tests[:5] + ["assert 1 == 1"]),
            ("def f(x):\n    return x", tests[:5] + ["assert f(1) == 1; import os"]),
            ("def f(x):\n    return x", tests[:5] + [7]),
        ]:
            with self.assertRaises(ValueError, msg=(bad_code, bad_tests)):
                a.check_answer("f", bad_code, bad_tests)

    # ---- stage 3 (local interpreter instead of Docker)

    def test_runner_reports_each_test(self):
        code = "def f(x):\n    print('noise')\n    return x + 1"
        tests = [
            "assert f(1) == 2",
            "assert f(1) == 3",
            "assert f('a') == 2",
            "raise SystemExit",
        ]
        self.assertEqual(
            v.run_tests(LOCAL_RUNNER, code, tests),
            [None, "AssertionError", "TypeError", "SystemExit"],
        )
        self.assertIsNone(v.run_tests(LOCAL_RUNNER, "import os\nos._exit(3)", tests))

    def test_fix_questions_need_a_test_that_catches_the_bug(self):
        answer = {"code": "def f(x):\n    return abs(x)", "tests": ["assert f(2) == 2"]}
        question = {"form": "fix", "code": "def f(x):\n    return x"}
        self.assertFalse(v.judge(LOCAL_RUNNER, question, answer)["passed"])
        answer["tests"].append("assert f(-2) == 2")
        self.assertTrue(v.judge(LOCAL_RUNNER, question, answer)["passed"])

    def test_container_gets_no_network_or_writable_disk(self):
        mock.patch.stopall()
        command = v.docker_command(["sudo", "docker"], "img")
        for flag in (
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--user=65534:65534",
        ):
            self.assertIn(flag, command)
        self.assertEqual(command[:3], ["sudo", "docker", "run"])
        self.assertLess(command.index("img"), command.index("timeout"))

    # ---- all three together

    def test_full_run_retries_one_failure_and_exports(self):
        self.quietly(q.collect, FakeClient(fake_questions), self.args)
        questions = c.read_jsonl(self.root / "questions.jsonl")
        unlucky = questions[0]["entry_point"]
        client = FakeClient(fake_answers(wrong_first={unlucky}))

        self.quietly(a.collect, client, self.args)
        self.assertEqual(len(client.requests), 40)
        self.quietly(a.collect, client, self.args)  # Waiting for verify: no new calls.
        self.assertEqual(len(client.requests), 40)

        self.quietly(v.verify, self.args)
        report = c.load_json(self.root / "report.json")
        self.assertEqual(
            (report["accepted"], report["counts"]["failed, one retry left"]), (39, 1)
        )

        self.quietly(a.collect, client, self.args)  # Only the failure is retried.
        self.assertEqual(len(client.requests), 41)
        retry = client.requests[-1]["messages"][1]["content"]
        self.assertIn("was rejected: AssertionError in test:", retry)
        self.quietly(v.verify, self.args)

        rows = c.read_jsonl(self.root / "accepted.jsonl")
        self.assertEqual(len(rows), 40)
        row = next(row for row in rows if row["entry_point"] == unlucky)
        self.assertEqual(
            (row["attempt"], row["response"]),
            (1, f"def {unlucky}(x):\n    return x + 1"),
        )
        self.assertEqual(
            set(rows[0])
            >= {"instruction", "response", "tests", "skill", "form", "level"},
            True,
        )
        report = c.load_json(self.root / "report.json")
        self.assertEqual(report["counts"]["accepted"], 40)
        self.assertGreater(report["usd_per_accepted"], 0)

    def test_two_failures_end_the_question(self):
        self.args.questions = 10
        self.quietly(q.collect, FakeClient(fake_questions), self.args)
        always_wrong = FakeClient(
            lambda prompt: {
                "code": "def "
                + re.search(r"must\s+define `(\w+)`", prompt).group(1)
                + "(x):\n    return 0",
                "tests": [
                    f"assert {re.search(r'must\s+define `(\w+)`', prompt).group(1)}({n}) == {n + 1}"
                    for n in range(6)
                ],
            }
        )
        for _ in range(3):
            self.quietly(a.collect, always_wrong, self.args)
            self.quietly(v.verify, self.args)
        self.assertEqual(len(always_wrong.requests), 20)
        self.assertEqual(c.read_jsonl(self.root / "accepted.jsonl"), [])
        self.assertEqual(c.load_json(self.root / "report.json")["counts"]["failed"], 10)

    def test_command_line_dry_runs_and_budget_exit(self):
        text = self.quietly(
            q.main, ["--questions", "30", "--dry-run", "--output", str(self.root)]
        )
        self.assertIn("3 calls planned", text)
        self.assertIn("Skill for this batch:", text)
        self.assertFalse((self.root / "requests").exists())
        self.args.budget_usd = 0.001
        with (
            mock.patch.object(q, "make_client", lambda: FakeClient(fake_questions)),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = q.main(
                [
                    "--questions",
                    "10",
                    "--budget-usd",
                    "0.001",
                    "--output",
                    str(self.root),
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
