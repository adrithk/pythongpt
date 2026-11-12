# DeepSeek dataset generator

Implemented November 12, 2026, at the user's request. The entry point is
deepseek_distill.py. It creates a new Python function dataset separately from S1.
No training, package installation, model download or paid generation runs on
import. Upload is an explicit separate command.

## Setup

Use the project's Python 3.12 environment and existing training dependencies:

~~~bash
source .venv/bin/activate
python -m pip install -r requirements-distill.txt
docker pull python:3.12-slim
python deepseek_distill.py check
~~~

Docker must be installed, running and usable by your terminal user. The script
never pulls images automatically. The check command runs a tiny trusted Docker
smoke test and loads only the local tokenizer at models/qwen-base. It does not
load model weights or call an API. On Windows use WSL2/Linux.

Provide DEEPSEEK_API_KEY in your environment or enter it at the hidden prompt.
The openai package is used as a compatible client pointed to DeepSeek; it does
not use an OpenAI API key. Do not put a literal key in source control.

## Generate and resume

Preview without dependencies, file writes, Docker or network:

~~~bash
python deepseek_distill.py run --candidates 100 --budget-usd 5 --dry-run
~~~

Example paid pilot with an illustrative USD 5 total accounting ceiling:

~~~bash
python deepseek_distill.py run --candidates 100 --budget-usd 5
python deepseek_distill.py status
~~~

DeepSeek chooses categories, weighted coverage and short briefs. Each brief
becomes a fixed specification. Fresh calls generate a checker without seeing a
candidate answer, a solution, and a quality/duplicate review. One solution repair
is allowed by default; the specification and checker stay fixed.

The model ID is deepseek-flash. Effort defaults to high; compare low/high on a
pilot before choosing a bulk setting. Teacher output budget defaults to 8192
tokens, while student admission defaults to 2048 combined Qwen tokens.

Data and SQLite state live under data/deepseek-distill. Candidate count is the
cumulative number of completed attempts including rejected examples, not a
guaranteed accepted count. Omitting --candidates targets 30000 attempts. The
pilot does not replace the larger SFT dataset plan.

Repeat a command to resume. Increase --candidates to extend the same run and
--budget-usd to raise its total ceiling (not an additional allowance). Other
settings must match. Use a new --output directory for different settings,
tokenizer files, Docker image, exclusions or split policy. A process lock
prevents simultaneous writers.

Budget exhaustion exits with code 2; Ctrl+C exits with 130. Completed candidates
and cached valid responses are reused. Lost/ambiguous network requests retain
their full cost reservation conservatively.

## Accounting and verification

All stages/retries are recorded. Before calling the API the script reserves an
output allowance and conservative input estimate. Actual prompt/completion
usage, including reasoning, replaces the reservation when available. Defaults
use peak uncached rates checked November 12: USD 0.30/million input and USD
1.20/million output. Update --input-price and --output-price when rates change.
Cache/off-peak savings are not assumed. This is estimated accounting, not a
provider-enforced billing limit. Planning, separate questions, reviews and
repairs add calls beyond earlier two-pass cost examples.

Supported tasks are deterministic synchronous functions with JSON-compatible
arguments/results, no input mutation, no expected exceptions, and a restricted
standard-library subset. This first implementation generates code-only tasks;
explanation/debugging conversations and arbitrary applications are not included.

Defaults require eight distinct fixed expected-value cases, an independent
simple reference consistent with those values, and 32 deterministic randomized
input comparisons. The trusted harness, rather than generated assertions,
decides success and checks input mutation. Float tolerance is 1e-9. Configure
--min-tests and --fuzz-cases as needed.

Generated code runs only in Docker: no network, read-only filesystem, non-root,
dropped capabilities, limited memory/processes/CPU/time. The image is resolved
to its immutable local image ID. AST filtering and limited builtins are scope
checks, not a Python security proof. Teacher tests/references/reviews may share
mistakes; passing them is not proof of correctness. Inspect samples manually.

## Deduplication and reserved evaluations

Exact normalized instruction/AST hashes reject repeats. SQLite full-text
retrieval (or a lexical fallback) supplies nearby accepted problems to the
teacher for semantic review and related-family grouping. This is not exhaustive
embedding-based search and cannot guarantee novelty.

Supply already-local reserved prompts/solutions; no benchmark is downloaded:

~~~bash
python deepseek_distill.py run --candidates 100 --budget-usd 5 \
  --exclude-jsonl data/reserved-evaluations.jsonl
~~~

Repeat --exclude-jsonl for multiple files. Recognized row fields include
instruction/question/prompt and code/response/completion/canonical_solution, or
messages. HumanEval-style prompt plus canonical_solution is assembled for AST
matching. Reserved material is never sent to the teacher.

Without supplied files the manifest records decontamination=not_run. Exact/
lexical checks cannot certify absence of semantic contamination. Export can
audit additional corpora and removes an entire related family on a match.

## Export and upload

Export is offline. Choose a new release path each time:

~~~bash
python deepseek_distill.py export \
  --release data/deepseek-distill/releases/v1
~~~

Add --exclude-jsonl with actual local files for an overlap-audited export.
Existing release folders are never overwritten. Artifacts include:

- source.jsonl with accepted records, checkers, provenance and split assignments.
- train.jsonl, validation.jsonl, test.jsonl with only prompt/completion targets.
- sft-splits/ containing a saved three-split Hugging Face DatasetDict.
- assignments.jsonl, rejected.jsonl, excluded.jsonl and integrity manifest.json.
- hub/ with source and sft Parquet configurations and a dataset card.

The source Parquet metadata_json field preserves heterogeneous test arguments.
It is not training text. Teacher reasoning remains in raw API files outside the
release. Qwen export uses a user message, assistant generation header, the
accepted code and one <|im_end|>, with complete student-token length checks.

Family-hash splits are 90/5/5 train/validation/test in expectation; tiny releases
can have empty holdouts. Related families stay together. Test should remain
untouched during development.

The existing init_trainsft.py still uses S1 paths, a 512-token limit and its
historical test-as-validation name. A later S2 trainer must explicitly use the
new paths, validation split and intended context limit.

After authenticating Hugging Face with its CLI or token mechanism:

~~~bash
python deepseek_distill.py upload \
  --release data/deepseek-distill/releases/v1 \
  --repo-id YOUR_USERNAME/python-distill-v1
~~~

Repositories are private by default; add --public only for intended public
publication. Existing visibility must match. Hashes and unexpected files are
checked before uploading only hub/. Raw API files, SQLite state, credentials and
original evaluation corpora are not uploaded. The script does not invent a
dataset license. Uploaded configurations are source and sft (default).

## Offline tests

~~~bash
python -m unittest discover -s tests -p 'test_deepseek_distill.py'
~~~

Tests use fake API responses, trusted authored code fixtures, mocked Docker
commands and real local Datasets/Arrow exports. Local-tokenizer checks run when
its files exist. No API calls or downloads occur. Real DeepSeek compatibility,
generation quality and Docker execution still require a PC pilot.

Sources:
[API](https://api-docs.deepseek.com/),
[JSON](https://api-docs.deepseek.com/guides/json_mode/),
[thinking](https://api-docs.deepseek.com/guides/thinking_mode/),
[pricing](https://api-docs.deepseek.com/quick_start/pricing/).
