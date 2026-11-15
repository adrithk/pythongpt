# Three-stage distillation

Written November 14, 2026. This replaces `deepseek_distill.py` as the way to
build the DeepSeek training set. The old collector and `deepseek_pipeline.py`
are untouched and still work.

Status: all offline tests pass. **No paid API call and no real Docker run has
been made with these files yet.** The first real run is the 200-question pilot
below.

## Why it changed

The first live pilot (November 13) showed four problems with the collector:

| Problem seen in the pilot | What the new design does |
| --- | --- |
| DeepSeek chose 20 topics and the script cycled through them | The program owns a list of about 235 skills and decides the mix itself |
| Questions were long rule lists; one burned 8,192 tokens of reasoning and returned nothing | Prompts ask for one short, clear function and forbid input-checking rules |
| One call at a time; about $0.005–0.01 each | Questions come 10 per call; calls run in parallel |
| Nothing was checked | Every solution runs against tests in Docker; only passes are kept |

## The flow

~~~text
distill_skills.txt        about 235 skills, one per line (edit it freely)
        |
distill_questions.py      program picks skill + form + difficulty,
        |                 DeepSeek writes 10 questions per call
        v
   questions.jsonl
        |
distill_answers.py        DeepSeek writes a solution AND 6-10 tests per question
        v
   answers/*.json
        |
distill_verify.py         runs solution against tests in Docker (no API cost)
        v
   accepted.jsonl         only rows whose tests all passed
~~~

`distill_common.py` holds what all three share: saving files, the spending
ceiling, the API call, and running jobs in parallel.

## Where the variety comes from

Asking a model to "be diverse" gives repeats, so the mix is set in Python, in
`plan_batch` in `distill_questions.py`:

- **Skill.** The skill list is shuffled once. Each batch of 10 questions gets
  the next skill, so all skills are used before any repeats.
- **Form.** Each question is assigned one of three forms by weight
  (`FORMS` at the top of the file):
  - `describe` (60%): a plain request, "Write a function ...".
  - `complete` (25%): a signature and docstring to finish. This is the shape
    your HumanEval+ benchmark prompt uses.
  - `fix` (15%): a buggy function to correct.
- **Difficulty.** easy 35%, medium 45%, hard 20% (`LEVELS`).
- **Repeats.** On later passes the prompt lists the function names already
  used for that skill. A new question that shares most of its words with an
  earlier one (same skill or same function name) is rejected.

To change the dataset, change those weights or edit `distill_skills.txt`.

## What "verified" means here

Before any code runs, the stages reject (by reading the code, not running it):

- a `complete` stub that already has a body. After the docstring only `pass`
  or `...` may follow. Its docstring must also show at least one `>>>`
  example;
- a test that does not call the function. `assert f is not None` mentions `f`
  but checks nothing;
- for `complete` and `fix` questions, a solution whose arguments or return
  type hint differ from the code the question showed.

`distill_verify.py` then keeps a row only if:

1. the solution runs and every one of its own tests passes within 10 seconds;
2. for `complete` questions, it also passes the `>>>` examples in the stub's
   docstring. These were written with the question, before any solution
   existed, so they cannot have been bent to fit the solution;
3. for a retry, it also passes the test that rejected the earlier attempt;
4. for `fix` questions, at least one test fails on the original buggy code.

A failed row gets one fresh attempt, which is told why the first was rejected.
After two failures the question is dropped.

Rule 3 drops some good rows on purpose. A retry writes new tests, and without
this rule it could get through simply by leaving out the test that caught the
first attempt. But sometimes that earlier test was the wrong part, and then a
correct retry is thrown away too. The retry prompt does not mention the rule,
so the model is not pushed to bend a correct solution to fit a wrong test. Set
`KEEP_FAILED_TEST = False` at the top of `distill_verify.py` to switch it off.

Limits you should know:

- The same model writes the solution and its tests, so a shared
  misunderstanding can pass. This still holds for `describe` and `fix`
  questions. `complete` rows also have the question-time `>>>` examples, but
  those come from the same model too, and there are only a few of them. Read a
  sample by hand.
- The repeat check compares words. It does not catch the same problem told
  with different words.
- There is no benchmark-overlap audit here. `deepseek_pipeline.py` has one
  (`--exclude-jsonl`).

## Setup

Stages 1 and 2 only need the API client. Stage 3 needs Docker, which is on the
PC and not on the Mac. `data/` is not in Git, so the simplest plan is to run
all three stages on the PC.

~~~bash
source .venv/bin/activate
~~~

~~~bash
python -m pip install -r requirements-distill.txt
~~~

~~~bash
docker pull python:3.12-slim
~~~

The API key is read from `DEEPSEEK_API_KEY`, or typed at a hidden prompt.

## Run the 200-question pilot

Look at the plan and one full prompt without spending anything:

~~~bash
python distill_questions.py --questions 200 --dry-run
~~~

Write the questions (20 calls):

~~~bash
python distill_questions.py --questions 200 --budget-usd 3
~~~

Solve them (one call per question):

~~~bash
python distill_answers.py --budget-usd 3
~~~

Run the tests. On the PC, Docker needed `sudo` for HumanEval grading, so:

~~~bash
python distill_verify.py --docker "sudo docker"
~~~

Give the failures their one retry, then test those:

~~~bash
python distill_answers.py --budget-usd 3
~~~

~~~bash
python distill_verify.py --docker "sudo docker"
~~~

Every command can be repeated safely. Finished work is skipped and saved
replies are reused for free.

About `--budget-usd`: it is one running total for the whole output folder,
across all stages. Each call first reserves its worst case (about $0.02) and
then replaces that with the real cost, so 8 parallel calls hold about $0.16
until their replies arrive. It is the script's own accounting, not a limit
enforced by DeepSeek. Prices are at the top of `distill_common.py` (peak
rates, checked November 14; off-peak is half).

## What to check before scaling up

`distill_verify.py` prints and saves `report.json`:

- `accepted` out of `questions`: the pass rate.
- `usd_per_accepted`: multiply by your target size for a cost estimate.
- `counts`: results split by form and difficulty. A form with a low pass rate
  needs its prompt fixed before a big run.
- `failed_attempt_reasons`: why each failed attempt failed, counting every
  attempt, including a first attempt whose retry then passed (the test text is
  cut off, so all `AssertionError` failures count together). This shows how
  often each rule above fires.

Then read 20 rows of `accepted.jsonl` yourself. Are the questions clear? Are
they different from each other? Are the solutions the kind of code you want
the model to write?

To scale, raise `--questions` (and the budget) and repeat the same commands in
the same folder. `--workers` controls how many calls run at once; DeepSeek
allows far more than the default of 8.

## Output files

Everything is under `data/distill/` unless you pass `--output`.

| File | Contents |
| --- | --- |
| `questions.jsonl` | Kept questions: `id`, `skill`, `form`, `level`, `entry_point`, `instruction` |
| `answers/<id>_<attempt>.json` | One solution with its tests |
| `verified/<id>_<attempt>.json` | Pass or fail, with the reason |
| `accepted.jsonl` | Final rows: `instruction`, `response`, `tests`, plus labels |
| `report.json` | Counts and cost |
| `requests/` | Every raw API reply, including reasoning and token usage |

`accepted.jsonl` is not yet in Qwen training format. `curate_build.py` does
that step (see [CURATE.md](CURATE.md)): `instruction` becomes the user prompt,
and `response` plus the end marker becomes the completion, as in
`init_prepare.py`. It drops rows over 1,024 tokens, the current trainer cap.

## Offline tests

~~~bash
python -m unittest discover -s tests -p 'test_distill*.py'
~~~

These use a fake API client, and the local Python interpreter in place of
Docker, running only fixture code written in the test file.
