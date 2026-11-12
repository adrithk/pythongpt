# Comprehensive Python RL roadmap

Research reviewed November 10, 2026. This is an assistant-proposed plan for Qwen/Qwen2.5-Coder-0.5B, not a selected recipe or completed experiment. Preserve B0 → S1 → R1, followed by improved S2 and comprehensive R2. S2 lineage remains open. No code changes, training, downloads, or paid resources are implied by this document.

## Recommendation and limits

Build a strong, verified SFT parent, then use online reinforcement learning with verifiable rewards (RLVR): sample Python solutions, execute independent grader tests, and update toward correct solutions. Start with the GRPO family and a capability-matched curriculum. Compare against rejection-sampled SFT under declared budgets. There is no published recipe establishing the optimum for this exact checkpoint, SFT history, hardware, and task distribution. Choose the final recipe using development correctness and cost, then evaluate once on a reserved final test.

The highest-priority investments are task/test quality, useful exploration, representative evaluation, and rollout efficiency. Algorithm complexity is a later variable. Long reasoning, repository agents, learned reward models, and distributed training are separate experiments rather than prerequisites.

## Research and transfer limits

- [DeepCoder-1.5B-Preview](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview): execution-based coding RL from an already distilled reasoning model; uses difficulty filtering and GRPO modifications. Evidence that the approach works for a smaller coder, but 1.5B is larger than this project and its long-context recipe is not a validated 0.5B configuration.
- [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783): identifies GRPO response-length and reward-normalization biases. Motivates explicitly comparing loss normalization; its headline experiments concern larger math models.
- [DAPO](https://arxiv.org/abs/2503.14476): dynamic sampling, asymmetric clipping, token-level loss and overlong handling; demonstrated on 32B math, not proof that the complete package is best here. Selecting a DAPO loss in TRL does not implement all of DAPO.
- [Small-model Python verification-feedback study, May 2026](https://arxiv.org/html/2605.30478v1): examines Qwen3-0.6B and Llama3.2-1B with LoRA on MBPP. Unit-test feedback is useful; lint-only rewards can harm correctness. GRPO and GSPO do not consistently dominate each other. Different bases, small benchmark setting, heuristic reward choices, and inconsistencies between narrative and tables warrant caution; do not project its numerical gains onto this project.
- [Offline code post-training study, 2026](https://arxiv.org/html/2609.11956v1): includes Qwen2.5-Coder-0.5B trained from base on CodeNet; reports gains and severe collapse with prolonged training. Particularly relevant model family/size, but no direct comparison of our future S2 + online GRPO versus offline training. Its raw-base baseline and checkpoint selection do not establish superiority over strong SFT. Treat offline preference/reward training as a later efficiency comparison.
- [TAROT, February 2026](https://arxiv.org/html/2602.15449v1): studies curricula using test difficulty and reports dependence on starting capability. Supports testing easier-to-harder progression for weaker models, not copying its full procedure without validation.
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1): distilling strong teacher outputs is an important small-model strategy. The reported settings do not prove that distillation always beats RL or that results transfer to 0.5B Python tasks.

## What the environment means

For a function-writing task, the observation is the specification, signature and permitted public examples. The action is the generated Python response (internally a sequence of token choices). The execution environment runs extracted code on grader inputs. The reward is an outcome score; an episode ends after grading. A rollout records prompt, response, sampling information and result. The policy is the model being trained. A frozen reference, when enabled, constrains drift; it is distinct from the policy snapshot that generated a batch.

Example: a prompt requests a function that removes duplicates while preserving order. Eight sampled implementations receive rewards [0, 1, 0, 0, 1, 0, 0, 0]. The group mean is 0.25. With group centering and no reward standard-deviation scaling, successful responses have advantage +0.75 and failed responses −0.25. The gradient makes rewarded response tokens more likely relative to the others, subject to the chosen objective. These are sequence outcomes, not labels identifying the exact faulty line. Fresh rollouts are sampled as weights change.

The single-turn environment needs no robotics simulator or Gym API. Later, a repair environment can return a public traceback/test result and allow one or two revisions. Repository work requires files, tools, dependency environments and longer episodes; introduce it only when it matches the deployment goal and basic tasks are reliable.

## Milestones

### 1. Establish the parent and measurements

Status updated November 12: the saved SFT script keeps its CUDA selection; the later MPS override was removed November 10. November 11 PC screenshots verify S1 reload, a finite-parameter check, generation, and HumanEval/HumanEval+ pass@1 of 0.305/0.280. Final training metrics and recovery lineage remain unsupplied, and improvement over B0 is not established. Preserve S1, obtain the baseline comparison, do R1 as the learning exercise, then develop S2 under recorded lineage. The short initial check remains distinct from the screenshot-evidenced PC S1 run.

Before R2 freeze prompt format, extraction, Python version, decoding and output limits. Measure S2 on development tasks, including multiple samples to estimate per-task solvability. Use held-out task families, not random paraphrases of training tasks. Keep evaluation problems out of teacher data, SFT, RL, curriculum selection and reward design. Existing exact-completion separation does not establish semantic separation or benchmark cleanliness. Audit SFT as well as RL sources.

Use [EvalPlus](https://github.com/evalplus/evalplus) HumanEval+/MBPP+ for external reporting, plus independent fresh tasks representative of intended Python use. Reserve final suites until settings are chosen. If MBPP training tasks are used, audit identity and overlap with the MBPP+ evaluation subset rather than assuming names imply separation. Record unavoidable possible base-pretraining contamination.

### 2. Build a verified RL task bank

Each task needs task/family/source IDs, revisions, specification, entry point or stdin/stdout protocol, allowed dependencies, public examples, private grader tests, verified reference, comparison rules, and execution limits. Gold code and private tests never enter the prompt. Pass IDs/tests to the reward function through metadata.

Current prepared rows contain prompt/completion only: they are SFT data, not a ready execution-reward dataset. Recover provenance and add independently checked tests or use dedicated problem/test sources.

Candidates: [Prime Intellect verifiable coding problems](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems) and [TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified), filtered to Python, supported protocols, suitable length and difficulty. These contain competitive-programming material; dataset scale or a verified label does not make every task appropriate. Function tasks and stdin/stdout tasks need different adapters. Avoid mechanically rewriting one format into another without checking semantics.

Verify references, manually inspect a diverse sample, add independent edge/property tests, reject flaky or ambiguous tasks, and check that plausible wrong implementations fail. Cluster task families across sources before splitting. Proposed starting pool: curate 1,000–5,000 diverse verified tasks; expand toward 10,000–20,000 only when useful coverage and measured throughput justify it. These are planning ranges, not optimal counts or a restriction on the existing whole-subset SFT preparation.

### 3. Implement one trustworthy execution interface

The trainer sends task ID and extracted code to isolated CPU workers; workers return pass counts, outcome category, runtime and infrastructure status. GPU generation/training and CPU grading are separate bottlenecks. Keep source code/test storage outside the candidate's control; preferably send individual inputs to isolated code and compare returned values in a trusted external grader. Some complex functions require specialized serialization/checkers.

Generated code is untrusted: disposable isolated workers, no network/credentials/host mounts, no writable grader state, and process/memory/CPU/wall-time/output limits. A timeout-only subprocess is not sufficient isolation. Use hardened containers or a stronger isolation boundary appropriate to the host; a generic Docker invocation alone is not a complete isolation design. Infrastructure failures are excluded/retried rather than punished as wrong answers. Test the runner with known-correct, known-wrong, crash, infinite-loop and resource-exhaustion cases before training.

### 4. Establish a transparent GRPO baseline

Initial task reward: 1 if all grader tests pass; 0 for wrong code, invalid response, crashes and candidate timeouts. Log partial pass rates and error categories. Add a partial-test curriculum only as a separate experiment if binary rewards remain too sparse after task selection. Do not reward lint, verbosity, specific reasoning phrases, or superficial syntax as substitutes for correctness. Performance rewards require reliable measurement and correctness gating.

Start by sampling 4–8 answers per problem. All-pass/all-fail groups have zero relative correctness advantage, though an enabled KL penalty may still contribute. Estimate task success with several samples; provisionally emphasize roughly 10–80% solvable tasks, retain coverage across skills, and periodically reassess. A zero in eight samples is an uncertain estimate, not proof of impossibility. Retain harder tasks for later and use verified teacher SFT to address missing skills. Bound any resampling/rejection budget so curriculum selection cannot consume the entire run.

Proposed pilot, not a validated optimal configuration:

| Setting | Starting point / comparison |
| --- | --- |
| Parent | frozen, evaluated S2 checkpoint lineage |
| Loss | Dr. GRPO; compare DAPO normalization |
| Reward scaling | no within-group std scaling; group centering remains |
| Samples per prompt | 8; 4 if rollout cost requires |
| Distinct prompts per optimizer update | target 8 initially, using accumulation |
| Full-weight LR | 1e-6; one comparison near 3e-6 if warranted |
| LoRA fallback | rank 32, separate LR pilot around 1e-5 |
| Sampling | temperature 0.8, top-p 1.0 |
| Completion cap | initially 512–1,024 tokens; assess completeness |
| Prompt cap | enough for complete specification; reject overlong rather than silently truncate |
| KL to frozen S2 | baseline beta 0.02; compare 0 if useful |
| Clipping | start symmetric 0.2; asymmetric upper 0.28 is later candidate |
| Rollout reuse | one iteration initially |
| Truncated responses | mask from training loss initially; still count as failures in evaluation and track rate |
| Budget | profile a representative 50–100 updates; scale by measured utility/cost |

With eight prompts and eight samples, an optimizer update uses 64 completions; at average 512 generated tokens this is about 32,768 rollout tokens before prompt processing, training and verification. A microbatch need not hold all 64 in memory. TRL batch sizes include sampled completions: verify actual distinct prompt count and group divisibility. The installed TRL 0.25.1 source supports dr_grpo/dapo, scale_rewards="none", epsilon_high and truncated-completion masking. Selecting a loss alone does not implement a full paper's sampling system. Do not copy latest-version API examples into the pinned environment without checking them.

Use full-weight training if the profiled run fits; LoRA is a memory comparison, not an established quality winner or loser. The proposed full-weight and adapter learning rates are different experiments. Masking excessive truncation can eliminate most useful learning; improve the cap/task distribution instead of silently accepting this.

### 5. Select the recipe with bounded comparisons

From the same S2, compare unchanged S2, rejection-sampled SFT on verified model successes, and the GRPO baseline. Report both sampling and optimization budgets; rerun the promising recipe with another seed where feasible. Next compare loss normalization or KL, one change at a time. Test verified teacher SFT and concise reasoning before assuming longer online RL is the highest-value investment. Any SFT after RL creates a new checkpoint lineage and requires evaluation for regressions.

Keep pass@1 (single attempt correctness) primary. Track pass@k separately for exploration, since more diversity or more attempts can improve pass@k without improving one-shot quality. Also track mixed-group fraction, truncation, response length, entropy, per-skill accuracy, infrastructure errors, reference divergence when enabled, throughput and cost. Use problem-level paired uncertainty estimates. Stop or revise when training reward rises without development gains, diversity collapses, the runner is unreliable, or the budget is reached. Select the best development checkpoint, not automatically the last checkpoint.

### 6. Final run and optional broader environments

Allocate the final run only after profiling. Historical cloud budget is approximately $100 for the whole later experiment, not authorization to spend. Preserve funds for final evaluation, independent tests and retries. Estimate total time as generation + grading + optimization + evaluation + saving; extrapolate measured pipeline timing rather than GPU hourly pricing alone. Log attempted and retained rollout tokens so filtering overhead is visible.

Mac: editing, preparation and inspecting results. Existing RX 9070 XT PC: intended local pilot target, subject to validated training support; start with Transformers generation. [AMD's WSL matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html) is version-specific and points toward newer unified documentation; recheck a matched stack at implementation time. An accelerated engine's general ROCm support does not establish support for this card/OS combination. Cloud NVIDIA is an optional practical path if verified rollout throughput or compatibility limits local work. No current price or hardware fit is guaranteed here.

Once single-turn R2 improves held-out correctness, test a separate repair stage with public feedback and private final tests. Report one-attempt versus fixed-budget repair separately. Add package tasks and repository environments only with corresponding target evaluations. Keep hidden final tests out of all tool feedback.

## Implementation order for hands-on follow-up

### Provisional RX 9070 XT runtime estimate — November 10

No matching measured benchmark or local RL throughput is available. Planning assumption only: with a working mixed-precision ROCm setup, eight distinct prompts × eight samples, average 512 generated tokens, and parallel CPU grading, allow approximately 2–5 minutes per optimizer update including generation, optimization and amortized evaluation. This is an unvalidated timing hypothesis, not a prediction derived from GPU specifications. It gives 50 updates ≈ 2–4 hours, 500 ≈ 17–42 hours, and 1,000 ≈ 33–83 hours. Plan about 2–4 days for one substantial 1,000-update run, and 1–2 weeks of machine time for a bounded set of comparisons/evaluation. Setup and task preparation are additional. Longer responses, serial generation, float32, resampling and slow graders can extend this considerably. Measure 20–50 representative updates before scheduling a long run; update count is not an established optimum or a guarantee of improvement.

Write small pieces in this order: task record → trusted isolated grader → generation/evaluation wrapper → task-difficulty report → reward callback → GRPO configuration → logging/checkpoint selection. Do not begin by copying a complete training script. Preserve tokenizer/chat template/end markers from the evaluated parent. Record model/data revisions, task IDs, software versions, configuration, seed, raw generations, checkpoints, compute, and cost for every run.
