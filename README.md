# PyDrithGPT

Current context and handoff updated **November 6, 2026**. Read the current-status sections before the reference roadmap. Inspect the actual saved files before advising on the next code change; they may have advanced since this snapshot.

## Project intent and agreed scope

Build a small model specializing in Python coding by post-training **`Qwen/Qwen2.5-Coder-0.5B` base**. The intended scale is hundreds of millions of parameters, not hundreds of billions. The long-term ambition is strong performance against similarly sized models, while retaining useful general assistant/instruction-following behavior.

The immediate goal is to **understand and complete a working SFT run**, save/reload the result, and establish a personal baseline. Basic execution-reward RL can follow afterward. There is **no first-run requirement to match Qwen Instruct**, reach a particular benchmark score, or implement advanced techniques. The Instruct checkpoint is a possible comparison, not the chosen starting checkpoint.

The user ultimately chose **base + Python-focused mixed SFT**, rather than starting from Instruct. A mix with general instructions, such as Smol-SmolTalk, remains a later experiment. **For the current first implementation, the user explicitly chose only OpenCoder stage 2's `educational_instruct` subset.** Do not silently reintroduce general data or change the starting model.

The pipeline must process the **whole selected subset**, not train on one preview example or a hard-coded 16/32-example dataset. The user initially requested a small learning run, then clarified that the code should support the whole dataset. Separate dataset preparation from run duration: later, discuss an epoch versus a temporary update limit as a configuration choice. No final training duration or settings have been agreed yet.

Here, “whole dataset” means all rows of `educational_instruct`, not automatically all four stage-2 configurations. Other configurations are `evol_instruct`, `mceval_instruct`, and `package_instruct`. Their possible later use has been discussed; none is currently included. Stage 1/stage 2 are names from OpenCoder's recipe, not mandatory stages for this Qwen experiment.

## Collaboration instructions for a new chat

- **Teach one small step at a time.** Explain what the code does, why it is needed, what the user should change, and the expected result.
- The user wants to **write the code and run experiments themselves**. Put small snippets in chat; do not generate full training scripts, install packages, download models, run training, or edit files by default.
- Reading the current code to check progress is welcome. When a specific edit is explicitly requested, perform only that edit. This README update is specifically authorized; it does not authorize implementation work elsewhere.
- Do not overwhelm the user with an entire file or many future implementation steps when they ask for the next step. A short overview is useful when requested, followed by one concrete action.
- The user is learning Python alongside SFT. Explain lists, dictionaries, function parameters, return values, methods, iteration, and terminal commands where relevant. Do not assume the distinction between Python code and terminal commands is obvious.
- When explaining an edit, identify exactly what to keep, remove, and replace in the saved file. Unsaved VS Code changes may differ from disk.
- Do not restart setup, benchmarking, or unrelated theory every turn. Preserve progress. Do not interpret inspecting one row as a decision to train only on that row.
- An earlier assistant overstepped by creating a complete starter implementation and downloading dependencies/model files. The user requested its removal; it was removed. Respect the hands-on workflow going forward.

## Hardware, environments, and budget

- Development machine: **M5 MacBook Air, 8-core GPU, 16 GB unified memory**, inspected locally. The user writes code, prepares data, and reviews results on this Mac.
- Main intended local training machine: Windows PC with **AMD RX 9070 XT, 16 GB VRAM**. The user prefers WSL2/Ubuntu over installing native Linux. WSL/ROCm training has not been validated in this chat.
- A first training experiment on the Mac's PyTorch MPS backend has also been discussed. Do not assume the user has definitively switched training to the Mac or that a full-parameter run fits its available memory; confirm the target before device-specific trainer settings.
- A later final cloud run should aim to stay below approximately **$100**. This is not a requirement to spend cloud money now or a guarantee of a competitive score.
- The intended project environment is **`.venv`**, using Python **3.12.14** when inspected. `.venvv` was an accidentally duplicated environment; the user was advised to keep `.venv` and remove the duplicate. Do not assume its deletion without checking.
- Last environment inspection found all shared requirements installed and no dependency conflicts. PyTorch **2.15.0** was installed in both environments at that time; core library imports succeeded in `.venv`. This is a historical observation, not a current training compatibility test.
- [requirements.txt](requirements.txt) pins shared packages: Transformers 4.57.3, TRL 0.25.1, Datasets 4.4.1, Accelerate 1.12.0, and Hugging Face Hub 0.36.0. PyTorch is installed separately for each platform. It can also be installed indirectly as a dependency; omitting it from this file does not prevent that.
- Recreate environments on the PC; do not copy the Mac's `.venv`. Install the appropriate AMD PyTorch build inside WSL, then shared requirements, and verify the installed build still supports the GPU.
- GitHub repository setup was completed by the user. Source/configuration belong in Git; model weights, data, caches, virtual environments, and training outputs are excluded by `.gitignore`. Do not commit or push without instruction.

## Current saved implementation

The project has been separated into **[init_prepare.py](init_prepare.py)** for preparation, **[init_inspect.py](init_inspect.py)** for checks on saved data, and **[init_trainsft.py](init_trainsft.py)** as an empty training placeholder. The former `tiny_initial_sft.py` was renamed to `init_prepare.py`; use the new names. Existing overlap-check code was moved into the inspection file, with loading from disk, rather than rebuilding preparation on every check. The user reports downloading the original model into **`models/qwen-base`** using Hugging Face CLI.

As of this snapshot, the saved script:

1. Imports `load_dataset` and `AutoTokenizer`.
2. Loads `OpenCoder-LLM/opc-sft-stage2`, configuration `educational_instruct`, split `train`, **without streaming**, and prints the full example count.
3. Loads the tokenizer from `models/qwen-base` with `local_files_only=True`.
4. Sets `tokenizer.eos_token` to Qwen's existing **`<|im_end|>`** marker.
5. Defines `format_example(example)`. It formats the row's **`instruction`** into a user prompt using `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`. It creates a completion from the row's **`code`** plus the ending token, then returns a dictionary containing `prompt` and `completion`.
6. Calls `coding.map(format_example, remove_columns=coding.column_names)` to create **`prepared` for every row**. It prints the prepared example count and column names.
7. Saves `prepared` to `data/prepared-sft`, randomly splits with `test_size=0.01` and `seed=42`, and saves the resulting training/validation datasets to `data/sft-splits`. The library's `test` key is being used for validation.

The preparation script currently prepares **text strings**, not training tensors. It does not load model weights, create a trainer, update weights, save a trained checkpoint, or execute the dataset's Python solutions/tests. The user has run preparation and reported **129 distinct identical completion strings shared across training and validation**. This is a known unresolved issue with the current random split. `init_inspect.py` reads the saved splits and reports that count without redoing preparation. `init_trainsft.py` is intentionally empty. Do not report that SFT has run or that the overlap has been fixed.

Raw source fields observed by the user are `seq_id`, `instruction`, `output`, `code`, `entry_point`, and `testcase`. The first preview was a palindrome function. `output` can include explanation/formatted code; the current target uses `code` alone. Dropping source columns from `prepared` does not mutate the original `coding` object. Retain provenance/split IDs when expanding the pipeline.

## Conversation-format explanation already established

The user was confused about the missing explicit assistant dictionary in the new function. Explain the current prompt/completion design consistently:

- **Prompt:** the user request wrapped in Qwen's template, ending with the assistant's starting header. `add_generation_prompt=True` adds that header; it does not generate an answer.
- **Completion:** the reference Python solution followed by `<|im_end|>`.
- Concatenating prompt and completion forms the complete conversation. The intended SFT loss applies to completion tokens only, including the ending marker.
- `tokenize=False` returns readable text. Tokenization into token IDs comes later. A previously inspected single full conversation contained **97 tokens**; that count was only an illustration, not a training configuration or dataset size.
- `format_example` defines the per-row transformation; `map` invokes it for all rows. `example` is the function parameter, not a global variable selecting a single training example.
- `next(iter(coding))` used earlier returned the first row, not a random sample. It was a preview and has been removed from the current formatting path.
- The user identified a long startup delay **between script start and completion of imports**, and accepted it. Do not keep redirecting the lesson to network/progress debugging unless a new issue occurs.
- Editor formatting-on-save and unused-import warnings were discussed. Neither is evidence of failed package installation. Save VS Code edits before checking the on-disk file.

## Next actions toward working SFT

Continue from the saved formatting code, rather than replacing it with a full new script:

1. Address the known 129 shared completion strings: teach the user how to group identical solutions before splitting so a solution's examples stay on one side. Keep all rows if possible, preserve reproducibility, and avoid mistaking 129 distinct solutions for 129 total overlapping examples.
2. After the user implements the split correction and reruns preparation, use `init_inspect.py` to verify zero exact completion overlap. This does not establish that all semantic duplicates have been removed. Preserve the selected data and source identifiers.
3. Choose a token-length policy and inspect lengths. Avoid silently cutting off target solutions. Keep the distinction between formatted strings, token IDs, and training labels clear.
4. Confirm the training machine/device, then load the original base weights. For the installed TRL release, use prompt-completion data and explicit **completion-only loss**. Check the actual prompt/answer masking and stopping token on one batch before a full run; inspection is not a dataset-size restriction.
5. Select learning rate, batch size, gradient accumulation, precision, memory-saving settings, and epoch/update budget together with the user. Earlier 4-update, 16-example, 1,000-example, and larger-dataset proposals are **not current fixed requirements**.
6. Record a simple baseline, run SFT, inspect loss/runtime/memory, and save the model, tokenizer, configuration, dataset versions/split IDs, and results separately from the original base.
7. Reload the trained checkpoint in a fresh process and compare held-out responses. Validation loss alone is not executable coding correctness. A low score does not invalidate the goal of understanding the process.

Basic RL, full benchmark suites, advanced curation, teacher distillation, dataset mixtures, and staged training are future work. The user asked about these and received conceptual explanations, but they should not block the first working SFT pipeline.

## Reference roadmap: earlier proposals, not immediate requirements

The notes below preserve longer-term research ideas from the initial planning discussion. Their example counts, hyperparameters, hardware assumptions, and benchmark plans are **provisional references**. They must not override the current scope and collaboration instructions above. [FIRST_SFT.md](FIRST_SFT.md) is also an older walkthrough; its tiny dataset sizes and “scripts do not exist” status text are superseded by this handoff. Recheck time-sensitive software support and cloud prices before using those notes.

## Goal and comparison rules

The starting model has approximately **490 million total parameters**, including about **360 million non-embedding parameters**. The Instruct checkpoint is Qwen's post-trained version. The technical report describes SFT followed by offline direct preference optimization (DPO). DPO is preference training; our proposed online RL stage is a separate experiment. Sources: [base model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B), [Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), [technical report](https://arxiv.org/html/2409.12186v3).

First target: finish the v0 SFT + RL experiment and establish our own measured baseline. Longer-term target: compete with the best models we can reproduce in a declared size category and inference budget. Qwen Instruct is a comparison point, not a milestone or acceptance criterion. Beating a 2024 checkpoint alone does not establish current state of the art.

Use total parameters, including embeddings, for size comparisons. Maintain a primary 300–600M parameter category and a separate broader sub-1B comparison. Qwen3-0.6B is a nearby comparator; Qwen3.5-0.8B belongs in the broader comparison. Their model cards establish availability, not their ranking on our suite: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).

When competitive benchmarking becomes the focus, choose the benchmark suite and define what “close” means before running the final comparison. Publish per-benchmark scores, paired uncertainty estimates, and the exact comparator set/date. No competitive-score threshold applies to v0.

Initial capability: generate a correct Python function from a specification, signature, and any public examples. Start with standard-library tasks. Broaden later to code repair, package use, and repository edits, with separate evaluations for each.

## Hardware and budget

- Development: the current Mac for writing code, preparing data/configurations, and reviewing results.
- Local training: Radeon RX 9070 XT, 16 GB VRAM, in the Windows PC. Prefer WSL2 with Ubuntu; native Linux is a fallback if measured compatibility or performance warrants it.
- Cloud: one final bounded experiment with a total budget of approximately **$100**. Do development and small ablations locally.

AMD lists the RX 9070 XT in its WSL compatibility matrix and documents ROCm GPU workloads through ROCDXG. Use an explicitly supported driver/runtime/PyTorch combination, then pin the working environment. Do not combine unrelated versions from different guides. Sources: [WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html), [AMD WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html).

The first hardware check must include a forward/backward pass, optimizer updates, checkpoint save/reload, and generation. GPU detection alone is insufficient. Record speed and peak VRAM over a representative 100-step training pilot. Keep project files and datasets inside the WSL Linux filesystem for the local workflow.

Start with ordinary PyTorch, Transformers, Datasets, Accelerate, and TRL. Use supported mixed precision; begin with FP16 on the documented WSL path and validate BF16 before enabling it. Start with eager execution or PyTorch attention that passes the backward check. Add specialized kernels or an accelerated rollout engine only after the ordinary implementation works. Avoid making quantization or an NVIDIA-specific optimizer a requirement.

Try full-parameter SFT with a microbatch of 1, short sequences, and gradient checkpointing. A rough optimizer/model-state allowance is 6–8 GB, depending on precision and optimizer representation, before activations, logits, runtime allocations, and generation caches. Thus 16 GB is a plausible development target, not a verified fit. Fall back to unquantized LoRA if needed.

AMD currently documents resets in some Radeon fine-tuning workloads and a hipBLASLt workaround. Consult that issue if a matching crash occurs; do not apply unrelated tuning preemptively. [AMD PyTorch known issues](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html#known-issues).

For cloud planning, Runpod's pricing page, updated November 4, 2026, lists a 24 GB RTX 4090 at $0.89/hour and 48 GB A40/A6000 options at $0.59/hour. Availability and actual checkout pricing must be rechecked. Compare cost per completed experiment; hourly price alone does not determine value. [Provider pricing](https://www.runpod.io/pricing).

Suggested spending ceilings, not runtime estimates:

| Component | Maximum allocation |
| --- | ---: |
| Final SFT | $15 |
| RL rollouts and optimization | $50 |
| Evaluation | $15 |
| Storage, setup, retries, and reserve | $20 |
| Total | $100 |

At $0.89/hour, $80 of GPU compute buys about 90 GPU-hours before other charges. This makes a bounded 0.5B experiment plausible. It does not guarantee a specific number of RL updates or a competitive final score. CPU test execution and rollout throughput may dominate.

Estimate SFT time from total processed tokens divided by measured training tokens/second. Estimate RL time from a pilot that includes generation, grading, reference scoring, and optimization. Include idle/setup/evaluation time in the billed total. Configure checkpointing and a wall-time ceiling before the final cloud job. Reserve cleanup time before the spending ceiling. Paid teacher generation would be a separate cost unless explicitly allocated inside this budget.

## Experiment sequence

| ID | Starting checkpoint | Treatment | What it measures |
| --- | --- | --- | --- |
| B0 | Qwen2.5-Coder-0.5B | None | Starting completion capability |
| B1, optional for v0 | Qwen2.5-Coder-0.5B-Instruct | None | Official post-training reference |
| S1 | B0 | Python SFT | Our supervised-training contribution |
| R1 | S1 | Execution-reward GRPO | Added value of online RL |
| C1, later | S1 | SFT on passing sampled solutions | Simpler alternative to RL |
| S2, optional | B1 | Same Python SFT | Best practical starting point for specialization |

Run B0, S1, and a short R1 first. B1 can be measured for context whenever convenient. Later, add C1 before making a strong claim that RL is worth its added cost. Use a recorded rollout budget for C1 and R1, and report differences in optimizer compute. A checkpoint trained from Instruct must be labeled as such; it is a different starting point from independently post-training the base.

## Stage 0: evaluation before training

For v0, a small fixed set of unseen Python problems with tests is enough. Measure the base, SFT, and SFT + RL checkpoints on it, and keep a separate development set for debugging and checkpoint choices. The broader public benchmarks below are later extensions and do not block the first training run.

Use HumanEval+ and MBPP+ through EvalPlus for early comparisons. EvalPlus adds substantially more tests than the original benchmark suites. Preserve original benchmark scores alongside the stricter scores. [EvalPlus](https://github.com/evalplus/evalplus).

Add BigCodeBench-Instruct to measure more practical function and library use. It is a different task format from BigCodeBench-Complete. Do not compare their raw scores as if they were the same test. [BigCodeBench](https://github.com/bigcode-project/bigcodebench).

Add a pinned, date-defined LiveCodeBench slice as a harder generalization check. Record release dates and audit overlap with post-training sources, including any teacher-generated data. A task being newer than the base model alone does not prove it is absent from our post-training data. [LiveCodeBench](https://livecodebench.github.io/).

For v0, the primary metric is single-attempt correctness on our fixed held-out set. For later public benchmarking, use HumanEval+ as an initial primary metric and MBPP+ as a secondary check. Before the competitive comparison, freeze the complete suite and report every component; do not choose the headline benchmark after observing the results. Track practical tasks separately from contest-style tasks.

Evaluation protocol:

1. Freeze problem IDs, benchmark revisions, dependency versions, Python version, prompt templates, extraction rules, stopping rules, and execution limits.
2. Use completion prompts for B0 and the documented chat template for instruction checkpoints. Keep the task information equivalent. Base/instruct comparisons measure the complete usable systems, while S1/R1 comparisons should share the same prompt format.
3. Start with greedy, one-completion evaluation for Qwen2.5 checkpoints. For other architectures, predeclare compatible decoding settings; report thinking modes and output-token budgets separately. Do not force an incompatible decoding mode to claim fairness.
4. Report one-attempt performance without execution feedback, plus generated tokens, truncation rate, formatting failures, syntax errors, timeouts, and latency. Use a common token cap for the primary comparison and separately show model-recommended settings if useful.
5. Later estimate sampled pass@1 and pass@k from fixed-temperature samples. Best-of-N with test-based selection or repair is a separate system and compute budget.
6. Select checkpoints using a separate development set. Run the final benchmark comparison after the recipe is frozen. Repeatedly consulting public tests turns them into development data.
7. Save problem-level outcomes and use paired bootstrap intervals over problems. Repeat finalist training seeds locally where feasible, or disclose a single-seed result. HumanEval has only 164 problems: a few changed answers are a small and noisy basis for a broad claim.

Official historical scores are useful checks, not guaranteed reproduction targets. Rerun B1 in our own environment before using it as the denominator of an improvement claim.

## Stage 1: build a small verified dataset

Start with the `educational_instruct` subset of `OpenCoder-LLM/opc-sft-stage2`. Its dataset card lists an MIT license and exposes instruction, code, entry point, and test cases. It is a concrete initial source to audit and filter, not a guarantee of clean data or sufficient diversity. [OpenCoder dataset](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2).

KodCode-V1 is a later candidate with paired solutions and tests, but its card lists CC BY-NC 4.0. Keep that source separate and account for its stated restrictions before using it. [KodCode dataset card](https://huggingface.co/datasets/KodCode/KodCode-V1).

Later expansion targets, after completing the smaller v0 dataset described above:

- 20,000 SFT training problems, with one concise correct Python solution per problem.
- 2,000–5,000 additional RL training problems with tests, where enough suitable problems survive filtering.
- Approximately 1,000 development and 1,000 internal test problems, grouped by source/seed/problem family before splitting.

Treat these counts as experiment sizes, not promised available rows. If deduplication leaves fewer examples, start smaller. Record total input and supervised output tokens as well as row counts.

Each record should retain: source and revision, original problem ID, family/seed ID, license metadata, specification, signature, entry point, reference solution, tests, topic, sequence length, and verification result. Grader tests and reference answers must be excluded from the model prompt. Only explicitly designated public examples may appear there.

Data preparation:

1. Filter to Python, self-contained functions, clear signatures, and the initial dependency scope.
2. Cluster exact and near duplicates across all sources, including rewritten problems. Split entire clusters/families together. A random row split is insufficient.
3. Audit overlap against every evaluation suite using normalized text, token overlaps, code structure, and available provenance. Remove suspect matches and preserve the audit results. This reduces contamination risk; it cannot establish that the original base never saw a benchmark.
4. Execute reference solutions against their tests in isolation. Reject broken, flaky, contradictory, or unsupported cases.
5. Audit a diverse sample of specifications and tests manually. Reference code and tests generated together can agree while both being wrong. Add independent edge cases or property checks where useful.
6. Preserve complete examples within the context limit. Exclude overlong examples initially instead of silently truncating solutions.
7. Balance basic strings, collections, numeric operations, parsing, and modest algorithms. Avoid letting many rewrites of one template dominate the training set.

## Stage 2: basic SFT

SFT teaches the model to produce the verified solution given the problem. Begin with code-only responses and short, explicit specifications. Long reasoning traces are an experimental variable to introduce later.

Provisional starting recipe, to be revised using local pilots:

| Setting | Initial choice |
| --- | --- |
| Training | Full parameters if memory permits; otherwise unquantized LoRA |
| Sequence length | 2,048 total tokens |
| Local microbatch | 1 |
| Effective batch | 32 examples through gradient accumulation |
| Full-training learning rate | 2e-5; small local comparisons at 1e-5 and 5e-5 if useful |
| LoRA fallback | Rank 32, LR around 1e-4; separately labeled experiment |
| Epochs | Start with 1; choose checkpoints on development correctness |
| Warmup | About 3% of optimizer steps |
| Precision | Validated mixed precision for the installed ROCm stack |
| Loss | Completion tokens only, including the intended end-of-answer token |
| Memory | Gradient checkpointing; generation cache disabled during training |

Use the Qwen Instruct chat format consistently for SFT and generation, with a verified tokenizer/template configuration and matching end tokens. Inspect a tokenized batch to verify that prompt tokens are masked and answer tokens receive loss. Do not assume a chat template's assistant-mask support without checking it.

TRL provides SFTTrainer and prompt-completion loss masking. Pin a tested release rather than depending on an evolving default. [SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer).

First, deliberately overfit 32–64 clean examples to check the pipeline, then perform the small v0 SFT run. Expanding to the larger S1 dataset is optional future work. A falling training loss alone is not evidence of better coding. Keep S1 even if RL later regresses.

## Stage 3: basic RL with verifiable rewards

Initialize from S1. Use TRL's GRPOTrainer with a custom execution reward. GRPO compares sampled answers for the same prompt and avoids a separately trained value model. [GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer).

Initial loop:

1. Sample a training problem and generate four candidate solutions. Use stochastic sampling, provisionally temperature 0.8, with a short output cap such as 512 tokens.
2. Extract each candidate using a fixed rule and execute it against the problem's grader tests.
3. Assign reward 1 only when all valid tests pass; assign 0 for incorrect code, invalid formatting, syntax errors, timeouts, and resource-limit failures. Record infrastructure/runner failures separately and exclude them from learning.
4. Apply small GRPO updates, starting near LR 1e-6. Explicitly choose the loss variant and a small nonzero KL penalty to a frozen S1 reference, provisionally beta 0.02. These are tuning hypotheses, not known optimal settings.
5. Evaluate held-out development correctness after short intervals, and retain the best development checkpoint within the compute cap.

For v0, run approximately 50–100 updates on easy tasks that S1 sometimes solves, adjusting the cap to measured runtime. Start with four candidates and small prompt batches; reduce generation concurrency before abandoning the experiment for memory reasons. On the AMD setup, use ordinary Transformers generation first. If rollout costs or compatibility become limiting, move R1 to cloud NVIDIA hardware.

With a binary outcome reward, groups in which every candidate fails have no relative correctness signal; groups in which every candidate passes also provide none. Track the fraction of groups with mixed outcomes. If almost all groups fail, use easier tasks or improve SFT. More RL updates alone will not fix an unusable reward signal. This diagnostic follows the group-relative advantage described in the TRL documentation.

Also track completion length, truncation, entropy, reference divergence, runner error rate, reward, and held-out accuracy. Do not reward verbosity or formatting independently of correctness in the first experiment. Training reward increasing while held-out correctness falls is a reason to stop and investigate.

Generated code is untrusted. Execute it in disposable isolated workers with no network, credentials, host mounts, or writable grader files, plus process, CPU, memory, and wall-time limits. Run the trusted grader separately enough that candidate code cannot rewrite the tests or simply report its own success. A bare subprocess and timeout are insufficient isolation.

## Stage 4: improve one variable at a time

Prioritize these experiments after the basic pipeline works:

1. **Data quality and coverage:** inspect failures, strengthen tests, remove duplicates, and fill missing Python skills.
2. **Verified teacher distillation:** generate better training solutions with a stronger model, keep independently verified answers, and compare concise solutions against reasoning-plus-code. Keep teacher costs and provenance explicit.
3. **Rejection-sampled SFT:** sample candidate solutions from S1, retain verified successes, and train on them. Compare C1 against R1 under declared sampling and training budgets.
4. **RL task selection:** emphasize tasks with mixed pass/fail outcomes, increase difficulty as capability grows, and test for generalization to unseen families.
5. **Repair training:** add buggy-code-plus-feedback examples and separately measure single-shot generation versus iterative repair with execution feedback.
6. **Optional DPO:** train on verified passing/failing pairs if they contain useful distinctions. Do not add it merely because Qwen used it, and do not call it online test-reward RL.
7. **Broader Python work:** introduce package APIs, longer contexts, and repository tasks once the corresponding evaluation and execution environments are reliable.

Continued pretraining is a later option only if evidence points to missing foundational knowledge. If substantial clean data and careful experiments plateau, compare a newer or slightly larger base before investing in a complicated algorithm. Post-training changes a fixed model's capabilities but offers no guarantee of reaching a chosen leaderboard score.

## Implementation boundaries for the next phase

Keep the eventual project small and modular:

- `data`: ingestion, provenance, deduplication, contamination audit, family splits, verification.
- `training`: SFT and GRPO entry points driven by saved configurations.
- `execution`: one isolated grading interface shared by data verification and RL, with evaluation workers kept separate.
- `evaluation`: benchmark adapters, generation, extraction, per-problem scoring, aggregate reports.
- `experiments`: immutable run manifests, configurations, metrics, and analysis.

Every run should record the model/tokenizer revision, parent checkpoint, dataset fingerprint and split IDs, software lock, configuration, seed, prompt/extraction versions, token counts, hardware, wall time, and billed cost. Retain raw generations and problem-level results. Store model weights and large datasets outside version control.

Immediate milestones:

1. Validate the WSL/ROCm training environment and checkpoint round-trip.
2. Verify the small v0 dataset, freeze separate development/test groups, and measure B0.
3. Complete the overfit check and the first basic SFT run; save S1.
4. Complete a short R1 run with informative execution rewards.
5. Compare B0, S1, and R1, then save a reloadable v0 model and a reproducible experiment record. This completes the first milestone regardless of its ranking against other models.
6. Use v0 as the baseline for later data, training, and benchmark experiments. Add B1 and broader public benchmarks as comparison points.
7. For a later larger cloud run, profile the recipe first and keep it within the approximately $100 envelope.
