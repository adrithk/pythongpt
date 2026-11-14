# Curating the S2 training set

Written November 14, 2026. Three small scripts turn four sources into one
training set of about 240K examples in the format `init_trainsft.py` loads.

Status: offline tests pass, and the collect and build steps were tried on a few
hundred real rows on the Mac. **The Docker step has not been run for real, no
full dataset has been pulled, and nothing has been trained on the result.**

## The flow

~~~text
curate_collect.py <source>    pull a source, convert every row to one shape,
        |                     clean the code, reject out-of-scope rows
        v
data/curated/candidates/<source>.jsonl
        |
curate_verify.py <source>     re-run each row's tests in Docker
        v
data/curated/verified/<source>.jsonl
        |
curate_build.py               format, drop long rows, drop repeats, drop
        |                     benchmark lookalikes, take quotas, split
        v
data/s2-splits                what the trainer loads
~~~

Every row, from every source, has the same shape:

~~~json
{
  "id": "educational:660113403",
  "source": "educational",
  "instruction": "Write a function to check if ...",
  "response": "def is_palindrome(s):\n    ...",
  "tests": ["assert is_palindrome(\"racecar\") == True"]
}
~~~

`response` is always code only. This is the same shape as the distill stages'
`accepted.jsonl`, so your DeepSeek rows go through the same steps.

## The sources

| Name | Dataset | Quota | Kept in a 400-row trial |
| --- | --- | ---: | --- |
| `opencode` | `nvidia/OpenCodeInstruct` (about 5M rows) | 120,000 | 52 of 400 |
| `educational` | `OpenCoder-LLM/opc-sft-stage2`, `educational_instruct` | 50,000 | 399 of 400 |
| `selfoss` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 40,000 | 348 of 400 |
| `deepseek` | your `data/distill/accepted.jsonl` | 30,000 | not tried |

The trial read only the first 400 rows of each source, so treat those rates as
a rough guide. Quotas are the `QUOTAS` dictionary at the top of
`curate_build.py`. If a source has fewer good rows than its quota, the build
takes what there is and the report shows the shortfall.

## What each step checks

**Collect** (no code is run):

- Pulls the code out of Markdown answers and keeps only imports, functions,
  classes and constants. Example calls and prints at the end are dropped.
- Rejects code that imports anything outside the standard library, or modules
  that reach outside the function (`os`, `random`, `time` and others in
  `BLOCKED_MODULES`), or that calls `input`, `open`, `eval` and similar.
- OpenCodeInstruct only: requires that every recorded test passed and every
  judge score is 5 (`MIN_JUDGE`). This is strict on purpose; it is why only
  about 1 row in 8 survives. Lower `MIN_JUDGE` to 4 if you need more rows.

**Verify** (in Docker, no network, time limit per row):

- Keeps a row only if its code loads and every test passes.
- Rows with no tests are kept if the code loads. Most `selfoss` rows are like
  this, so that source rests mainly on its own upstream filtering.

**Build**:

- Formats each row exactly like `init_prepare.py` and drops rows over 1,024
  tokens.
- Drops exact repeats (same instruction, or same code ignoring spacing).
- Drops near repeats: same function name and mostly the same words.
- Allows at most 30 rows per function name (`NAME_CAP`).
- Drops any row whose function name matches a HumanEval+ or MBPP+ task, or
  whose wording mostly matches one. Benchmark text is used only for this check.
- Takes each source's quota at random, then holds out up to 1,000 tested rows
  as the validation split.

## Run it

On the PC, with the `.venv` active. The benchmark check needs `evalplus`
(`requirements-eval.txt`), which you already used for grading.

Try a source first without saving anything:

~~~bash
python curate_collect.py opencode --peek 400
~~~

Pull each source. `--take` stops once that many rows have been kept; for
OpenCodeInstruct that means reading well over a million rows, so it takes a
while.

~~~bash
python curate_collect.py educational
~~~

~~~bash
python curate_collect.py selfoss
~~~

~~~bash
python curate_collect.py opencode --take 160000
~~~

~~~bash
python curate_collect.py deepseek
~~~

Re-run the tests for each source (repeatable; it resumes where it stopped):

~~~bash
python curate_verify.py educational --docker "sudo docker"
~~~

Repeat that command with `selfoss`, `opencode` and `deepseek`.

Build the set:

~~~bash
python curate_build.py --name s2
~~~

Then change one line at the top of `init_trainsft.py`:

~~~python
DATA = "data/s2-splits"
~~~

You can build before the DeepSeek run is finished: a missing source is simply
left out. Use a different `--name` for each build; an existing set is never
overwritten.

## Read the report before training

`curate_build.py` prints and saves `data/curated/s2-report.json`:

- `rows_per_source_at_each_step`: how many rows each step removed.
- `removed`: the reason counts, per source.
- `benchmark_check`: must say it ran. `NOT RUN` means the set is unchecked.
- `train_tokens_total`: the size of one epoch in tokens, for time estimates.
- `most_common_function_names`: if a few names dominate, lower `NAME_CAP`.

Also open `data/curated/s2-train.jsonl` and read a sample from each source.

## Limits

- Passing tests is not proof of correctness, and `selfoss` rows mostly have no
  tests at all.
- The repeat check compares words and function names. The same task under a
  different name and wording gets through.
- The benchmark check catches copied names and near-copied wording, not a
  benchmark task that was fully rewritten.
- All thresholds here (`SIMILAR`, `NAME_CAP`, `BENCHMARK_SIMILAR`, the quotas)
  are first guesses that have not been tuned against results.

## Offline tests

~~~bash
python -m unittest discover -s tests -p 'test_curate.py'
~~~
