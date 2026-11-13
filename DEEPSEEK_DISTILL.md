# Small DeepSeek collector

The previous 2,031-line implementation combined API generation with a database,
Docker verification, repairs, release auditing and upload. Those features remain
available in deepseek_pipeline.py; the standalone collector is deepseek_distill.py.

## What you need to understand

~~~text
DeepSeek chooses topics
  -> creates distinct short problem briefs
  -> writes a question and code for each brief
  -> Python parses the JSON into a dictionary
  -> checks syntax and exact repetitions
  -> saves the candidate dictionary to disk
~~~

Rows are explicitly unverified. Correctness checks, semantic/benchmark-overlap
review, grouped splits, Qwen formatting and upload are in the separate pipeline.

## The seven functions

| Function | Why it exists |
| --- | --- |
| save_json | Saves a whole file atomically so interrupted writes cannot destroy it. |
| fingerprint | Makes a hash for detecting changed requests and exact duplicates. |
| request_json | Makes the API call, saves raw replies, reuses cached replies, checks fields and enforces the accounting ceiling. |
| make_record | Checks syntax/function name and exact repeats, then builds one dictionary. |
| export_candidates | Writes JSONL: one serialized dictionary per line. |
| collect | The main generation loop: plan, brief, ask, check, save. |
| main | Reads terminal options, obtains the key, creates the client and calls collect. |

BudgetLimit stops the next request when its reservation would exceed the budget.
Constants at the top control the model, output allowance, batch size, prices and
generation instructions.

## Run one example

The collector needs the optional API client, not a GPU or local model:

~~~bash
source .venv/bin/activate
python -m pip install -r requirements-distill.txt
python deepseek_distill.py --candidates 1 --budget-usd 1 --dry-run
~~~

Remove --dry-run when you intend to make paid requests. USD 1 is an example
accounting ceiling. The script reads DEEPSEEK_API_KEY or asks at a hidden prompt.
Leave that prompt and api_key=key unchanged.

The first example normally needs three calls: topics, a batch of ten briefs,
and one question/answer. Future examples reuse topics/briefs. At most two new
attempts are made per job per invocation for malformed replies.

Data lives under data/deepseek-candidates. Repeat the command to resume; increase
--candidates to extend the collection. It counts attempts, including rejects,
rather than guaranteed usable examples. Old advanced-pipeline directories stay
separate. The budget is cumulative for this folder, including all stages and
retries. Interrupted requests retain reservations; actual usage includes
reasoning. This is conservative accounting, not a provider-enforced billing cap.
Verify top-of-file prices before bulk generation. The folder is locked to one
writer at a time.

## Dictionary format

~~~json
{
  "instruction": "Write a function ...",
  "response": "def example(...):\n    ...",
  "entry_point": "example",
  "status": "unverified"
}
~~~

Rows also retain IDs, topic, seed brief, provenance and duplicate hashes. Raw API
responses/usage are separate. Rejected rows have a reason.

A dictionary is the object in Python memory. JSON is its text serialization.
JSONL stores one serialized dictionary per line of candidates.jsonl.

## Verify and create the training dataset

Important checks were preserved in deepseek_pipeline.py. They consume the saved
questions/answers without paying to generate those initial answers again:

~~~bash
python deepseek_pipeline.py run \
  --import-candidates data/deepseek-candidates/candidates.jsonl \
  --output data/deepseek-verified --budget-usd 5

python deepseek_pipeline.py export \
  --output data/deepseek-verified \
  --release data/deepseek-verified/releases/v1
~~~

Docker, the local Qwen tokenizer and project training dependencies are needed
for this stage. Checkers/reviews and any required repairs cost additional API
usage. Add local --exclude-jsonl corpora for overlap audits. Passing generated
tests is not proof of correctness or absence of benchmark contamination.

The export creates Qwen prompt/completion with the assistant header and ending
token. Publication remains explicit, with private-by-default repositories and
integrity checks. Full setup and commands: [DEEPSEEK_PIPELINE.md](DEEPSEEK_PIPELINE.md).

## Offline tests

~~~bash
python -m unittest discover -s tests -p 'test_deepseek*.py'
~~~

Tests use fake API clients and preserve the original pipeline's tests. They do
not spend API credits, install packages, download models, run Docker or upload.
