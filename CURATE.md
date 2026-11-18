# Curating the S2 training set

Written November 14, 2026. Three small scripts turn three sources into one
training set of about 240K examples in the format `train_torch.py` loads.
(November 17: `educational_instruct`, S1's data, was dropped by user decision;
the OpenCodeInstruct quota went from 120,000 to 200,000.)

Status, November 17: the full pipeline ran on the Mac and built `data/s2-splits`
(135,406 training rows, 1,000 validation rows, 49.7M training tokens; report in
`data/curated/s2-report.json`). OpenCodeInstruct gave about 94K rows, not 200K:
most of its candidates were repeats. Nothing has been trained on it yet.
Later the same day all 5M OpenCodeInstruct rows were read (422,147 candidates)
and two more sets were built: `data/s2-full-splits` (cap 30, 180,029 training
rows) and `data/s2-200k-splits` (`--name-cap 100`, 197,374 training rows, 77.1M
tokens). The default cap stays 30; pass `--name-cap N` to change it.

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
  "id": "opencode:58193295bd590a8dc33b27c6b2846795",
  "source": "opencode",
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
| `opencode` | `nvidia/OpenCodeInstruct` (about 5M rows) | 200,000 | 9,024 of 100,000 (first file) |
| `selfoss` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 40,000 | 348 of 400 |
| `deepseek` | your `data/distill/accepted.jsonl` | 30,000 (takes all 3,071) | all verified already |

`educational` (`OpenCoder-LLM/opc-sft-stage2`, `educational_instruct`) still has
an adapter in `curate_collect.py`, but it is not in `QUOTAS`, so the build
ignores it.

The self-oss trial read only its first 400 rows, so treat that rate as a rough
guide. Quotas are the `QUOTAS` dictionary at the top of
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

- Formats each row exactly like `initial/init_prepare.py` and drops rows over 1,024
  tokens.
- Drops exact repeats (same instruction, or same code ignoring spacing).
- Drops near repeats: same function name and mostly the same words.
- Allows at most 30 rows per function name (`NAME_CAP`).
- Drops any row whose function name matches a HumanEval+ or MBPP+ task, or
  whose wording mostly matches one. Benchmark text is used only for this check.
- Takes each source's quota at random, then holds out up to 1,000 tested rows
  as the validation split.

## Run it

On the Mac (or the PC), with the `.venv` active. Nothing is saved until you run
these: `curate_collect.py` reads each source from Hugging Face over the network,
except OpenCodeInstruct, which it reads from your disk once you have downloaded it.
The benchmark check needs `evalplus` (`pip install -r requirements-eval.txt`);
it is not yet installed in the Mac's `.venv`. On the PC, Docker needed
`--docker "sudo docker"`; on the Mac leave that option off.

Try a source first without saving anything:

~~~bash
python curate_collect.py opencode --peek 400
~~~

Download OpenCodeInstruct once (50 files, about 6.9 GB, into `data/opencode-raw`;
files already there are skipped, so you can stop and rerun):

~~~bash
python load_opencode.py
~~~

Only about 9% of its rows pass the filters, so `--take 260000` (the 200,000
quota plus a cushion for rows the later steps drop) reads roughly 2.9 million
rows. From disk that takes a few minutes and draws from all 5 million
rows in a fully shuffled order; streaming (used only when no local files exist)
would take much longer and mixes only 10,000 rows at a time.

Pull each source. `--take` stops once that many rows have been kept.

~~~bash
python curate_collect.py selfoss
~~~

~~~bash
python curate_collect.py opencode --take 260000
~~~

~~~bash
python curate_collect.py deepseek
~~~

Re-run the tests for each source (repeatable; it resumes where it stopped):

~~~bash
python curate_verify.py opencode
~~~

Repeat that command with `selfoss` and `deepseek`.

Build the set:

~~~bash
python curate_build.py --name s2
~~~

Then change one line at the top of `train_torch.py`:

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
