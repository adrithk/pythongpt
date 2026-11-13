# Proposed DeepSeek-to-Qwen dataset pipeline

Current layout after the user's November 12 simplification request: the small,
independent collector is deepseek_distill.py. The full workflow described below
is preserved in deepseek_pipeline.py, with an import option that reuses collected
questions/answers. See DEEPSEEK_DISTILL.md for learning the collector and
DEEPSEEK_PIPELINE.md for verification/export/publication. Candidate collection
does not imply correctness or training readiness.

Design drafted November 11, 2026. On November 12, the user explicitly requested full code and deepseek_distill.py was implemented; see [DEEPSEEK_DISTILL.md](DEEPSEEK_DISTILL.md) for actual commands, supported scope and limits. This document retains the proposed methodology and is not evidence of a successful production run. Teacher-generated share, accepted yield, spending limit and S2 lineage remain open. A 30K-candidate tranche does not replace the larger SFT planning target in AGENTS.md.

## Evidence and scope

Public examples support teacher-generated data with selection and evaluation, rather than indiscriminately training on all responses. [OpenAI's distillation guidance](https://developers.openai.com/api/docs/guides/supervised-fine-tuning#distilling-from-a-larger-model) describes tuning the teacher prompt and selecting qualifying outputs. [Qwen2.5-Coder, sections 4–5](https://arxiv.org/html/2409.12186v3#S4) describes code-inspired synthesis, quality scoring, execution verification and decontamination. [Llama 3, sections 4.2–4.3](https://arxiv.org/html/2407.21783v3#S4) describes topic classification, quality/difficulty signals, semantic deduplication and code generation with execution feedback. These are published examples, not access to private current frontier-lab recipes. The concrete engineering choices below are project proposals, not a proven 0.5B optimum.

Initial target: self-contained Python standard-library functions, with explicit signatures and edge-case behavior. Keep complete, concise targets. Explanation and debugging can be added as explicitly labeled task types with appropriate responses and checks; do not strip requested explanations merely to retain code-only targets. Keep reserved benchmarks and their derivatives out of generation, training and reward development.

## 1. API boundary and durable jobs

Use the OpenAI-compatible client with a locally supplied DeepSeek key and `base_url="https://api.deepseek.com"`. The requested model is `deepseek-flash`; record both the requested alias and returned identifier. An alias is not a guaranteed immutable model revision.

Sources checked: [first call](https://api-docs.deepseek.com/), [JSON output](https://api-docs.deepseek.com/guides/json_mode/), [thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/).

Each stage constructs a prompt and requests JSON. Persist the full returned response and usage before parsing. Validate completion status, nonblank content, JSON object shape, required fields/types and stage-specific conditions. JSON syntax is not schema or correctness validation. Teacher reasoning is not the assistant training target.

Give each stage/attempt a host-assigned ID. Store its parent seed/problem, prompt version, actual request settings, timestamps, response ID, raw path, usage and outcome in SQLite or a durable manifest. Reuse saved responses on resume; an incomplete or rejected attempt gets a new bounded retry record. Network-ambiguous failures cannot be assumed unbilled. Keep API rate/concurrency settings configurable and begin sequentially.

Track all-stage cost, rejected candidates and repairs. Reserve headroom for in-flight requests when applying a spending cap. Do not extrapolate cost from visible code tokens alone or treat previous two-pass estimates as a full multi-stage budget.

## 2. Teacher-generated coverage plan

Ask the teacher for a structured set of categories, subskills and task forms appropriate for this project. The user does not need to write every topic. Store the plan and have the controller track both attempted and accepted coverage. Avoid assuming equal quotas, teacher difficulty labels or a particular row count are empirically optimal.

For a more grounded extension, draw seed inspiration from suitable licensed source snippets or vetted source problems, retaining their provenance. Benchmark questions are not seed material. Teacher-only briefs are a legitimate initial experiment, with their coverage and novelty still requiring an audit.

## 3. Distinct content briefs

Generate small batches of problem ideas guided by coverage gaps. Each brief names a core operation, input/output behavior and meaningful constraint. The host assigns seed IDs. Compare briefs to prior instructions and retrieve a few relevant near matches to discourage paraphrases. A random seed, unique ID, different variable names or a new story does not establish a new problem. Recurring algorithms can still support distinct useful exercises.

Distinguish broad category IDs, seed lineage and semantic duplicate groups. The whole strings category should not become one split group merely because it shares a label. Never assume that 30K briefs or API responses imply 30K distinct accepted tasks.

## 4. Freeze problem specifications

Expand each retained brief into one full user-facing instruction containing the signature, types, return behavior, constraints and relevant edge cases. Check that every behavior required by later tests is present in the instruction. Freeze and version this specification before solution generation. Record parent seed/source IDs. Reject ambiguous or out-of-scope problems before paying for further work.

## 5. Generate candidate solutions

Give the frozen problem to the teacher and request a complete concise solution. For initial function tasks, the code becomes the intended assistant response. Parse it, check the requested entry point and screen for scope/import issues before execution. Compilation and AST parsing do not prove correctness or provide a sandbox. Keep failed solutions as diagnostic artifacts; do not accept them as successful one-turn SFT targets.

## 6. Build verification evidence

In a separate fresh request, give the problem specification without the candidate solution and ask for tests and, where feasible, a simple reference implementation. Tests should cover stated behavior, boundaries and plausible wrong solutions. Add deterministic randomized/property comparisons where suitable. Check that assertions actually execute and exercise the named entry point. Sample manual review is still necessary; same-teacher solutions/tests/oracles can share mistakes.

Execute candidate and checker code in disposable restricted containers: no network or secrets, non-root user, bounded time/memory/process/output resources, and only required files. Record interpreter/container versions and outcomes. Test quality can be probed with deliberately wrong variants, such as ignoring duplicates when duplicates matter. Test passage is evidence against this checker, not proof of general correctness.

If a solution fails, allow a bounded repair using the fixed specification and failure details. Keep original tests fixed during solution repair. If a test/specification is wrong, revise it explicitly, version the change and rerun affected checks; do not silently weaken tests to accept code. Accept only after the configured checks and quality review pass. An exact number of retries/tests is a pilot choice, not a lab-established universal rule.

## 7. Final selection and group splits

Combine execution evidence with review of specification clarity, instruction following, response style and usefulness. Prefer explicit rejection reasons to one unexplained judge score. Recheck duplicates across all synthetic and existing sources using normalized text/code, AST/token similarity and semantic candidate review. Audit benchmark overlap independently of teacher claims of originality.

Keep related source descendants, paraphrases and problem variants in one split. Freeze development/test policies before iterative prompt or training tuning; use development feedback for design, leave the final test untouched. Assign/reconcile groups after full-corpus duplicate clustering so later duplicates cannot cross splits.

## 8. Canonical records and Qwen export

Keep canonical instruction/response plus test code, optional oracle code, seed/source/group IDs, model and prompt settings, token usage, verification evidence and rejection history. Store raw responses separately. JSONL is convenient while appending records; Parquet is suitable for a frozen Hub dataset. Define a consistent schema across rows and splits.

For accepted rows, derive `prompt` using the local Qwen tokenizer's user chat template and assistant generation header. Derive `completion` from the intended assistant response plus `<|im_end|>`. Generation scaffolding, JSON wrappers, judging text, tests and teacher reasoning stay outside training text. Preserve metadata in canonical artifacts instead of dropping it permanently.

Measure actual prompt+completion token lengths with the student tokenizer. The saved S1 trainer caps 512 combined tokens. A proposed 2048-token S2 admission/training cap needs an explicit setting change and memory/stability pilot; do not truncate correct answers or assume API output budgets equal student context lengths. Verify completion-only masking, EOS handling and no unwanted extra special tokens before a new run.

## 9. Freeze, upload, train and evaluate

Write an audit manifest: attempted/accepted/rejected counts by reason, unique groups, coverage, length histogram, measured tokens and teacher cost, split hashes, prompt/code versions and limitations. Freeze the accepted release, then export/upload a new versioned dataset. [Hugging Face upload](https://huggingface.co/docs/datasets/upload_dataset) supports a new private repository. Uploading is an artifact step, not evidence of training readiness or model quality.

Train under a recorded checkpoint lineage and compare under a consistent evaluation protocol. Same-base S2 is the proposed clean S1/S2 comparison, not a selected user decision. Check saved/reloaded weights and use development correctness plus untouched final evaluation; training loss alone does not establish improvement. Keep S1 and all benchmark artifacts intact.

## Suggested implementation order

1. One API call, raw caching, field validation and canonical candidate record.
2. One end-to-end example through sandbox verification and Qwen export.
3. A roughly 100-candidate pipeline/quality/cost audit, not a mandatory small SFT run.
4. Durable jobs, generated coverage plan/briefs, duplicate retrieval, bounded repairs and cost controls.
5. Scale to the selected candidate budget, audit the actual accepted release, then upload/train.

The implementation uses one entry point with focused API, planning, verification, curation and export functions, separate from the working S1 scripts. At the time of the November 11 proposal no generator had been written. November 12 implementation was checked offline without package installation, paid API calls, a real Docker run, upload or training; see the implementation guide and TRAINING_LOG.md.
