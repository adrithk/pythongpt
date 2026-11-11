# Agent guide for PyDrithGPT

Last reviewed: November 10, 2026. This is the shared, repository-local context for coding assistants, including assistants without chat history. Read this file first, then inspect the relevant saved code and TRAINING_LOG.md. README.md is intended for readers and may be rewritten without preserving agent instructions there.

## Required startup procedure

At the start of every new session in this repository, read this AGENTS.md before project analysis, recommendations, or edits. If the tool already supplies this file as repository instructions, use that supplied content. After a context reset or handoff, reread it when it is no longer available in context. Then read relevant TRAINING_LOG.md entries and inspect the files needed for the task.

If delegating work when authorized, include this startup requirement in the handoff and ensure the receiving agent has access to this file. Do not rely on private assistant memory or access to old chats. Follow the maintenance instructions below as project facts change.

## Keep this context current

- Update this file whenever work establishes a material decision, implementation change, blocker, or next action. The user authorizes ongoing maintenance of this agent context as part of project work.
- Append run observations and results to TRAINING_LOG.md; keep the current status and next actions here concise. Link larger plans/results rather than duplicating changing facts.
- Distinguish user decisions, inspected facts, user-reported results, and untested proposals. Date verification. Never mark a planned run complete from its configuration or from a prior assistant's claim.
- Replace obsolete current-status statements; retain consequential corrections and provenance. Keep historical run records intact.
- Before finishing project work, check whether these notes and the training log need an update. Do not invent progress when nothing changed.
- Keep CLAUDE.md as a short entry point to this file, so context has one authoritative home. Commit these documentation files when the user requests a commit; do not commit or push by default.
- Current user instructions take precedence over this guide. Actual saved files establish implementation state; inspect them before suggesting changes. Ask only for material missing information that cannot be found locally.

## Collaboration preferences

The user is learning Python and post-training and wants to write the code and run experiments themselves. Explain manageable snippets, where each goes, what to keep/remove/replace, why it works, and the expected output. A few small steps per response are welcome; avoid complete training-script dumps. Explain Python concepts and distinguish terminal commands from Python when relevant.

Read-only inspection is welcome. Make scoped implementation edits when asked; otherwise teach rather than silently implementing, installing packages, downloading models, or running training. Documentation maintenance is authorized. Do not restart setup or unrelated theory each turn. Unsaved editor contents may differ from disk. A prior assistant created an unsolicited full implementation and downloads, which the user asked to remove: preserve the hands-on workflow.

## Project purpose and fixed starting scope

Post-train Qwen/Qwen2.5-Coder-0.5B **base** for Python coding, initially self-contained standard-library functions. The target scale is hundreds of millions of parameters. Long-term goal: strong measured performance among comparable small models, with useful instruction following. Official Qwen Instruct is an optional comparator, not the starting checkpoint or a first-run acceptance threshold.

Initial data: OpenCoder-LLM/opc-sft-stage2, educational_instruct only. Prepare the whole selected subset; do not confuse a preview or short update budget with reducing preparation to 16/32 examples. Other OpenCoder configurations and general instruction mixtures remain later experiments. OpenCoder's stage names do not mandate stages for this project.

## Current saved implementation and progress

Inspected November 10, 2026:

- init_prepare.py loads the whole educational_instruct subset, formats instruction as a Qwen user prompt with the assistant generation header, and code plus `<|im_end|>` as completion. It saves data/prepared-sft and data/sft-splits. It initially splits 1% with seed 42, then moves validation rows whose exact completion appears in training into training. Source columns are removed from formatted rows; provenance retention is a future improvement.
- init_inspect.py loads saved splits, counts exact completion overlap, and inspects full prompt+completion token lengths. This does not audit semantic duplicates or benchmark contamination. The old README's unresolved 129-overlapping-solutions status is superseded by the implemented move logic; source chats reported zero exact overlap afterward. No fresh dataset inspection was run for this guide.
- Source chats reported 118,278 raw rows, 117,224 train / 1,054 validation after the move, and 116,874 train / 1,049 validation after the 512-token filter. These counts are historical observations, not freshly recomputed here.
- init_trainsft.py is implemented, not empty. It filters complete examples above 512 tokens, loads the local base in float32 with SDPA, disables generation cache during training, and uses SFTTrainer with completion-only loss, no packing, gradient checkpointing, adamw_torch, LR 2e-5, microbatch 1, accumulation 16, one epoch, seed 42, and evaluation/save every 500 steps. It saves training metrics and final model/tokenizer under runs/initial-sft. Configuration is not proof this run finished.
- **CUDA device fix, November 10:** user requested the PC device change and push. Removed the later MPS override from init_trainsft.py; the script now keeps its existing CUDA availability check and `torch.device("cuda")` selection for the PC's ROCm build. Python syntax was checked; no GPU training was run for this edit.
- Saved check weights exist at runs/initial-sft-check/final/model.safetensors. Its train_results.json records runtime 61.5814 seconds, loss 0.6273176551, epoch 0.0006844978; chats report five Mac MPS optimizer updates. This is a completed pipeline check, not evidence of broad coding improvement or a full PC run.
- No RL implementation/completion has been verified. The next intended local SFT precedes the tiny RL run. Preserve existing user edits; inspect git status before editing.

## Format and teaching continuity

The prompt is readable chat-template text ending with the assistant header; add_generation_prompt=True adds that header, not an answer. Completion is reference code plus the ending marker. Together they form the conversation; loss should cover completion tokens including the ending marker. tokenize=False produces text, not tensors. Verify actual masking and stop behavior when changing training setup.

format_example defines a per-row transformation; map applies it to every row. A past 97-token example was illustrative, not a dataset or context-size requirement. next(iter(dataset)) previews the first row, not a random sample. Import startup delay was previously accepted; editor formatting/unused-import warnings do not establish dependency failures.

## Machines, dependencies, and storage

- Development: M5 MacBook Air, 8-core GPU, 16 GB unified memory; saved check used MPS.
- Intended main local training: Windows PC, RX 9070 XT with 16 GB VRAM, preferably WSL2/Ubuntu. A working PC training environment and full run remain unverified here. Confirm actual target before device-specific guidance.
- Recreate .venv per machine; do not copy the Mac environment to the PC. Historical Python version: 3.12.14. Reinspect current dependencies rather than assuming old checks still hold. requirements.txt pins Transformers 4.57.3, TRL 0.25.1, Datasets 4.4.1, Accelerate 1.12.0, Hugging Face Hub 0.36.0; PyTorch is hardware-specific. Check installed-version APIs before giving trainer snippets.
- models/qwen-base is the original base. Keep trained checkpoints separate. Data, weights, caches, environments, and runs are excluded from ordinary Git; cloning source does not transfer them. Download the base on the PC and transfer exact prepared splits/checkpoints or recreate data with recorded revisions/checksums.
- Later cloud budget discussed: approximately $100 total, not current spending authorization. Recheck compatibility and pricing before acting. GPU detection alone does not validate backward passes, optimizer updates, generation, or checkpoint reload.

## Next actions and open choices

Continue the user's first local SFT work from saved code. The CUDA/MPS conflict was fixed on November 10 at the user's request. Confirm that the PC checkout has the fix, establish actual GPU/run/checkpoint status, preserve baseline outputs, complete/save/reload S1, and record results. Then teach the tiny RL exercise in manageable pieces. Do not impose a new large tuning search or repeated tiny checks once their learning purpose is fulfilled.

Later SFT data mixture, scale/token budget, checkpoint lineage, hyperparameters, and comprehensive RL recipe remain open. The clean comparison proposal is to train S1 and S2 from the same base; this is not a confirmed choice of S2 lineage. Keep validation for development and an untouched final test; the dataset key `test` currently means validation. See the experiment plan below for comparisons and controls.

## Experiment sequence and comparisons

1. B0: original Qwen base baseline.
2. S1: first local SFT; complete/save/reload/evaluate before tiny RL.
3. R1: S1 + simple practice RL. Sorting-list correctness and about five GRPO updates were proposed, not completed or fixed requirements. This demonstrates the loop, not broad coding gains.
4. S2: significantly larger, improved SFT after the first cycle; data curation, mixtures/distillation, and tuning remain research choices. Same-base training is a proposed comparison control; record actual lineage.
5. R2: S2 + comprehensive RL, researched later. GRPO is a provisional default, not an established optimum.

Compare B0/S1, S1/R1, S1/S2, R1/S2/R2, and S2/R2. Preserve all checkpoints. Use consistent held-out tasks, grader, prompt/decoding settings, and generation budgets. Separate sorting-task results from coding correctness. Record unequal compute, data/model revisions, seeds, runtime, cost, and lineage; loss alone does not establish improvement.

## Supporting documentation — read as relevant

- EXPERIMENT_PLAN.md: detailed current plan and historical research roadmap. Historical counts, recipes, budgets, and milestones are proposals; current scope above supersedes conflicting older suggestions.
- TRAINING_LOG.md: dated observations and run results; append new evidence without rewriting historical outcomes.
- FIRST_SFT.md: historical walkthrough; its 1,000-example proposal is superseded.
- README.md: reader-facing overview, with some historical status still awaiting cleanup. Prefer this guide and inspected code for current agent context.

Do not load the full historical roadmap unless the task needs it. Shared context should stay concise; move detailed research and history to the supporting files rather than expanding startup instructions.
