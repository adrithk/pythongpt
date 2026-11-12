# PyDrithGPT training log

This is the persistent record for training updates and results shared in the documentation chat. Created November 9, 2026.

## How this log is maintained

- Append dated updates as the user reports progress, runs, results, errors, or decisions.
- Record reported observations separately from interpretations and proposed next steps.
- Preserve earlier results; add corrections explicitly rather than silently replacing history.
- Record run dates when provided. Otherwise, use the date the update was received and mark the run date as unspecified.
- Leave missing details as unknown. Planned settings are not completed runs, and falling loss alone is not a measured improvement in coding ability.
- Keep large logs, datasets, and checkpoints outside this document; link to them when available.

## Run index

| Run | Status | Evidence |
| --- | --- | --- |
| initial-sft-check | Saved local check; source chat reports five Mac MPS updates | `runs/initial-sft-check/final/model.safetensors`; `runs/initial-sft-check/train_results.json` |
| initial-sft (PC S1) | Completion and saving reported November 11; reload/evaluation pending | Earlier screenshots verify GPU progress, resumption, and a late NaN incident; no final metrics or recovery outcome supplied |
| S1 → R1 → S2 → R2 | Planned experiment sequence | [Current plan and comparisons](README.md#current-experiment-plan--november-10-2026) |

PC SFT completion/saving is user-reported; final checkpoint integrity, generation, benchmark gains, and RL results have not been independently verified.

## Update history

### November 9, 2026 — Documentation started

**User request:** Use this chat to collect information about project training and its results, and maintain a persistent written record.

**Status:** Awaiting the first training update. No new training results reported.

## Details to capture when available

For each run, record its identifier and purpose, date, starting model/checkpoint, dataset and split, hardware and software environment, training settings, duration and cost, metrics and evaluations, saved artifact locations, errors, conclusions, and agreed next steps. Updates can be informal; only supplied or verified details should be filled in.

### November 10, 2026 — Experiment sequence clarified; earlier check verified

The user plans initial SFT on their computer before tiny simple RL, followed by significantly larger/improved SFT and later comprehensive RL after further research. Compare initial versus improved SFT, initial SFT + simple RL versus improved SFT versus improved SFT + comprehensive RL, and each RL checkpoint against its own SFT parent. See README for measurement controls and open recipe choices.

Verified locally today: the saved initial check weights exist; its results record runtime **61.5814 seconds**, train loss **0.6273176551**, and epoch **0.0006844978**. Run date remains unspecified here; the source chats reviewed on November 10 report five updates on Mac MPS. These are training-process observations, not measured coding gains. The intended next local SFT and subsequent RL stages remain planned. No final optimized recipe has been selected.

### November 10, 2026 — Comprehensive RL research roadmap proposed

User requested research and an explanation of optimal Python RL, environments, and implementation milestones. Reviewed saved preparation/SFT code, pinned and installed TRL 0.25.1 configuration source, and primary research including DeepCoder, Dr. GRPO, DAPO, TAROT, and 2026 small-model Python/offline code studies. Saved [RL_ROADMAP.md](RL_ROADMAP.md), with transfer limits and citations.

Assistant proposal: improved verified SFT followed by execution-reward GRPO-family training, capability-matched Python tasks, isolated grading, and comparisons against rejection-sampled SFT. Suggested hyperparameters/task counts are untested starting points, not a selected optimum. The exact-model offline study trains from base and does not settle our post-SFT online recipe. No implementation, installation, downloads, training, evaluation, or spending occurred. Existing device conflict remains. User experiment order and open S2 lineage remain unchanged.

Follow-up runtime question: saved an explicitly unvalidated RX 9070 XT planning estimate in RL_ROADMAP.md: approximately 2–4 days for a substantial 1,000-update run under the specified short-response/mixed-precision assumptions, with substantial uncertainty. No local RL speed measurement was obtained; this does not establish a required update budget or completed progress.

### November 10, 2026 — Missing dependency reported during setup

User reported that `generate-parameter-library-py 0.7.3` requires `typeguard`, which is not installed. The affected machine, active environment, and full command output were not supplied. Suggested installing the missing dependency in the affected project environment and running `python -m pip check`; no installation or successful resolution has been verified. This ROS-related package is not an established SFT project dependency; its visibility does not establish that it belongs in `requirements.txt`.

### November 10, 2026 — Torch import blocked by a missing shared library

User reports an import error while attempting the earlier Torch GPU verification snippet: a shared library transcribed as `librocx64` cannot be opened. The exact filename including `.so` suffix/version, full traceback, installed Torch/ROCm versions, and runtime search paths have not been provided. WSL PC setup is the conversation context, not a remotely inspected environment. Requested the exact import traceback and installed package/runtime information before selecting a fix. No runtime installation, path modification, successful GPU check, or training progress is established.

Follow-up evidence: user supplied `IMG_9122.JPG` showing the PC terminal. Verified visually: the missing filename is `libroctx64.so.4`; `pip show torch` reports `2.9.1+rocm7.2.1.lw.gitff65f5bc` in `/home/adrith/Developer/pythongpt/.venv/lib/python3.12/site-packages`; `/opt/rocm*` does not exist and the `/opt` library search prints nothing. This supports a missing standard ROCm runtime setup, rather than an SFT code failure; it does not rule out a nonstandard installation. Ubuntu release, WSL `/dev/dxg` availability, Windows driver, and successful runtime installation remain unknown. AMD's ROCm 7.2.1 WSL guide requires ROCm userspace and ROCDXG, in addition to the Python wheels. No fix has been executed or validated.

Further user-provided output confirms Ubuntu **24.04.4 LTS (Noble)** and an existing `/dev/dxg` character device. This establishes the distro and WSL interface, not functioning GPU compute. Verified primary-source installation references: AMD's ROCm 7.2.1 quick start registers its Noble repository and installs `rocm`; ROCDXG's v1.2.0 compatibility table includes ROCm 7.2.x and RX 9070 XT, and its published asset is `rocdxg-roct_1.2.0_amd64.deb`. Provided a WSL userspace-only installation path, ROCDXG runtime installation, temporary shell environment settings, and `rocminfo`/Torch verification commands for the user to execute. No installation or successful verification is yet reported; Windows driver compatibility remains to be established if compute initialization fails.

Next reported error: import now fails on a MIOpen shared library, transcribed as `libmiopen so 1` (likely `libMIOpen.so.1`; full new traceback not supplied). This is not proof that every preceding installation step succeeded or that GPU initialization works. AMD's ROCm 7.2.1 MIOpen documentation identifies `miopen-hip` as the Ubuntu package. Suggested installing that package from the configured matching repository, refreshing the loader/library path, and collecting unresolved dependencies with `ldd` if import still fails. No successful fix, GPU detection, or training run is verified.

Next supplied screenshot (`codex-clipboard-a3a449cb-abac-4824-8308-edc7dfc8d1fd.png`) shows import stopping on `libhipfft.so.0`. The user's `ldd` output lists `libhipfft.so.0`, `libhiprand.so.1`, `libhipsparse.so.4`, `libhipsolver.so.1`, `libhipsparselt.so.0`, and `librccl.so.1` as unresolved; `libhipsparse.so.4` appears twice. Suggested installing their corresponding ROCm packages together from the matching repository, refreshing the loader/path, and repeating the import/GPU check. Package installation outcomes and GPU detection remain unverified.

User then reported "ok worked now what to do". This is a reported recovery after the dependency-install/import/GPU-check instructions; no exact final console output was supplied. It does not independently verify that GPU availability was `True`, actual GPU computation succeeded, or any SFT run completed. Reinspected the Mac checkout: the CUDA check is present but the later `device = torch.device("mps")` override remains at line 41; full-epoch configuration and periodic validation are present. Guided the user to continue full SFT on the PC only with successful GPU detection, remove the conflicting device override in their copy, ensure model/data/dependencies exist, then launch training. No implementation edits or training were performed by the assistant.

### November 10, 2026 — CUDA device override fixed

User explicitly requested changing applicable device settings to CUDA and pushing the change. Inspected project Python files and found one conflicting later MPS assignment in init_trainsft.py; removed it, preserving the existing CUDA availability check and CUDA device selection. Verified Python syntax and reviewed the diff. No package installation, model download, GPU computation, or training was performed for this edit; the PC training run remains unverified.

### November 10, 2026 — PC S1 training started; early timing observed

User supplied `IMG_9125.JPG`. Visually verified: AMD Radeon RX 9070 XT, model device `cuda:0`, 116,874 training examples and 1,049 validation examples, logged training loss and gradient norms, and **65/7,305** optimizer updates completed. The progress bar reports **03:28 elapsed**, **06:49:03 remaining**, and **3.39 seconds/update**. This supersedes the earlier unverified GPU-computation status; actual PC GPU training is now evidenced by the supplied screenshot, not by remote execution.

Interpretation: approximately seven hours of training remain at the early displayed pace; periodic validation/checkpoint saving and changing batch lengths can alter the estimate. Planning around seven to eight hours is provisional, not a measured completion time. The first scheduled validation/checkpoint at update 500 is not yet shown. Full-run completion, final model saving/reload, and benchmark gains remain unverified.

### November 10, 2026 — First PC S1 validation observed

User supplied `codex-clipboard-285edb3c-1b15-4750-8d17-89a21844bb7a.png` (same displayed progress as the preceding interrupted screenshot). Visually verified: **587/7,305 updates**, **33:00 elapsed**, **06:15:58 estimated remaining**, and **3.36 s/update**. Recent logged training losses are approximately **0.41–0.49**; latest visible loss is **0.4581**. Logged gradient norms are finite, roughly 4–7 in the visible interval; no NaN/Inf or runtime error is visible.

First visible validation result: `eval_loss=0.4562495925712585`, `eval_runtime=66.9288`, `eval_samples_per_second=15.673`, `eval_mean_token_accuracy=0.8655486163265712`, epoch 0.07. This is consistent with update-500 evaluation under the saved configuration; the screenshot does not explicitly show that update number or confirm a saved checkpoint file. The accuracy measures supervised next-token prediction, not percentage of working Python solutions.

Assessment: training and validation loss are of similar scale, with no obvious instability in this snapshot. One validation result cannot establish an overfitting trend or improvement over the base. Continue the run and compare the next validation result near update 1,000 against 0.45625. Current remaining-time display is about 6 hours 16 minutes; future evaluation/save pauses and varying batch lengths can change it. Run completion, final checkpoint reload, and coding correctness remain unverified.

### November 10, 2026 — PC turned off during S1; checkpoint recovery pending

User reports "my computer turned off while training". The cause (sleep, shutdown, power loss, or crash), time, last completed update, and checkpoint availability/integrity have not been supplied. Last verified screenshot remains at update 587; a completed update-500 save is plausible from the configured schedule and subsequent progress, but no checkpoint directory has been inspected on the PC.

Reinspected saved code: output `runs/initial-sft`, save every 500 updates, retain two checkpoints, and `result = trainer.train()` without a resume argument. Verified installed Transformers 4.57.3 source: an explicit `resume_from_checkpoint` path restores available model/optimizer/scheduler state; `True` selects the highest-numbered directory and does not validate its completeness. Guided checking the original terminal after waking (sleep may preserve the process), otherwise listing PC checkpoints before choosing a complete one. No implementation change, resume, fresh restart, checkpoint deletion, or completed recovery has been performed or verified.

Follow-up user report: "checkpoint 5000 is there". This establishes reported directory availability, not checkpoint completeness or integrity. Guided checking for model weights, optimizer/scheduler state, and trainer state before replacing only the training call with explicit `resume_from_checkpoint="runs/initial-sft/checkpoint-5000"`. Keep the original `num_train_epochs=1`, dataset, and output directory. If valid, 5,000 updates (~68%) can be recovered and 2,305 remain; at earlier observed 3.3–3.4 seconds/update, remaining training is roughly 2.1–2.2 hours before validation/save overhead. No successful resumption or new model metric is verified.

Next evidence: `IMG_9128.JPG` shows the PC directory listing for `runs/initial-sft/checkpoint-5000/`, including model.safetensors (~1.9G), optimizer.pt (~3.7G), scheduler.pt, rng_state.pth, trainer_state.json (~147K), training_args.bin, and tokenizer/config files. These are nonempty expected checkpoint files; their deserialization/integrity is not verified by a listing. The terminal prompt does not visibly show an activated `.venv`. Provided environment activation/ROCm exports and a one-call resume edit for the user to make and run on the PC. No source implementation edit, git push, resumed training, or new loss measurement was performed/verified in this step.

### November 10, 2026 — PC S1 checkpoint recovery succeeded

Supplied `codex-clipboard-f9c5ea9c-a37b-4d60-a5fc-e93088dc8c5d.jpg` shows the user activating `.venv`, setting ROCm environment variables, editing `init_trainsft.py` with Nano, and launching training. Output names RX 9070 XT, model device `cuda:0`, 116,874 training / 1,049 validation examples, and **5,001/7,305** updates. This verifies visible recovery of the saved step count and a subsequent update; the resumed PC file itself was not remotely read and the Mac source remains unchanged. Full completion and new loss/validation results are not yet evidenced.

The output includes a missing checkpoint key warning for `lm_head.weight`. Checked local base and five-step saved configs: both have `tie_word_embeddings=true`; installed Qwen2 implementation declares `_tied_weights_keys=["lm_head.weight"]`. Read both local safetensors headers without loading tensors: input embedding weights are present and a separate `lm_head.weight` is absent. Official Qwen config also enables tied embeddings. The warning is consistent with shared embedding/output weights saved once, not evidence by itself of a lost trained output layer. No implementation change was made for it.

Initial resumed bar shows ~1,239.83 it/s and ~one second remaining after four seconds elapsed. This is a progress-counter artifact from jumping to the restored 5,001 count and must not be used for runtime planning. Wait for subsequent updates; earlier observed ~3.3 seconds/update suggests roughly 2–2.5 hours remain including provisional validation/save overhead. No new measured steady-state resumed speed has been supplied.

### November 10, 2026 — Final-loss interpretation after resume

User asked what final loss to target. No new run metric was supplied. Explained that there is no universal target: compare held-out `eval_loss` over the run, with the first observed ~0.45625 as a checkpoint reference, not an untrained-base baseline; validate generated-code correctness separately.

Inspected installed Transformers 4.57.3 `trainer.py`: `_total_loss_scalar` resets to zero after resumed TrainerState loading (around line 2568), while final `train_loss` divides that resumed-session loss sum by the full `state.global_step` (around lines 2815–2817). Thus final summary training loss can be underreported for this resumed run. Regular interval `loss` uses the steps since the previous log (around line 3205), and evaluation independently measures the held-out set. Suggested using interval/validation trends rather than treating an unusually low final summary as a quality gain. This is source-verified reporting behavior; no final run output or training-update failure is established, and no implementation change was made.

### November 10, 2026 — Persistent NaNs during resumed S1

User supplied `IMG_9130.JPG` asking whether it is a major issue. Visually verified: progress **5,683/7,305** (~78%), resumed-session elapsed **37:03**. Visible finite training losses ~0.30–0.33 and gradient norms ~3–4 are followed by loss **494.8906**, `grad_norm: nan`, `entropy: nan`, then repeated `loss: 0.0` with NaN gradients/entropy and collapsed token accuracy. Exact first failing update is not printed; learning-rate progression places it approximately around 5,630, an inference rather than a verified step identifier.

Advised immediate Ctrl+C and preserving existing checkpoints; actual stop is not confirmed. This is a real numerical failure, distinct from the final aggregate-loss reporting issue. Inspected pinned Trainer code: default `logging_nan_inf_filter=True` substitutes displayed loss values but does not prevent invalid gradient computation/application. Saved config is float32, fp16/bf16 disabled, SDPA, AdamW; default max_grad_norm is already 1.0. Root cause (data, numerical operation, backend, or another issue) is not established, so no new learning-rate/precision/backend prescription or implementation edit was made.

Recovery proposal: list and preserve PC checkpoints, inspect model and optimizer tensors/recent logs for finite state, select a pre-failure checkpoint, and use separate recovery output with fail-fast diagnostics before committing to another long continuation. Checkpoint 5,500 is a candidate if present and valid; 5,000 previously loaded and trained normally. Retention is two checkpoints, so continuing a poisoned run can rotate out earlier valid saves. Current checkpoint inventory, tensor integrity, successful recovery, and final results remain unverified. Remote Desktop Commander discovery was attempted for read-only access but returned Unknown tool; no PC command was executed through it.

Follow-up user reports checkpoint-5500 exists and, after checking its logs, says NaNs began afterward. This supports a pre-failure checkpoint candidate but is not an independent finite-tensor inspection. No diagnostic tensor scan was run. Proposed resumption from that explicit candidate with original data and training horizon, new output `runs/initial-sft-recovery-5500` to preserve original saves, `logging_nan_inf_filter=False`, and `torch.autograd.set_detect_anomaly(True)` before the train call. PyTorch 2.9 primary docs confirm NaN-producing backward operations can raise with a forward traceback under anomaly detection; it slows execution and does not guarantee detection of every possible optimizer/Inf issue. It is a diagnostic change, not a claimed remedy. No learning-rate/precision/attention change or new max_steps cap is proposed.

Inspected pinned Trainer resume handling: saved TrainerState can restore logging cadence after initial configuration; left `logging_steps` unchanged rather than promising per-update logging from only a config edit. No implementation edit, successful diagnostic continuation, new failure trace, or root-cause determination has been verified.


### November 11, 2026 — S1 completion reported; PC inference and HumanEval+ requested

User reports the model is done training and saved, and explicitly chooses PC inference and PC benchmarking. Wants simple problems and a LeetCode example before HumanEval+. No final console output, metrics, exact saved path, or explanation of recovery from the preceding NaN incident was supplied. Completion is user-reported; finite weights, successful reload, and coding improvement remain unverified.

Inspected current source: training still saves model/tokenizer to the configured output directory's `final` child; the local source has the original output directory, while the previously proposed recovery used `runs/initial-sft-recovery-5500`. Do not assume which exists on the PC. Prepared teaching scripts in `/tmp/pydrith-post-training` for the user to copy, rather than editing their training implementation: `infer.py` checks finite saved parameters and uses the training chat template, explicit Qwen stopping tokens, FP32/SDPA CUDA, and greedy generation; `benchmark_humaneval.py` generates one full-function answer per HumanEval+ task, saves raw/sanitized answers and metadata, and sends no reference/test data to the model; `grade_humaneval.py` checks matching dataset hashes and invokes EvalPlus inside Docker. Proposed host EvalPlus version is 0.3.1; inspected its APIs and the official container workflow. This is an initial same-chat comparison protocol, not a reproduction of Qwen's published scores or a contamination audit.

Validation here: Python parsing and a stub-model check using the locally saved Qwen tokenizer verified prompt formatting, answer slicing, stopping-token removal, and greedy configuration. Stub benchmark plumbing verified no reference/hidden-test leakage into prompts, solution-format output, metadata, and refusal to reuse an output directory. No PC checkpoint was read, no model inference/benchmark/training run was performed, and no packages, models, containers, or datasets were installed/downloaded by the assistant. Actual PC GPU generation, EvalPlus extraction, and Docker grading remain to be run by the user. No commit or push.

### November 11, 2026 — User-authored inference starter reviewed

User asked to inspect their VS Code `infer.py`, fix mistyping, and push. The saved file contains the walkthrough imports and `load_model()` only; no generation function or main/CLI has been saved. No Python typo or loading-logic error was found. Clarified the GPU-unavailable error message; preserved the user's incremental implementation and explained that running this starter alone does not generate answers.

Checks passed: Python parsing; mocked GPU-unavailable branch; mocked finite-weight loading, CUDA placement, and eval mode; rejection of NaN and Inf parameters. No packages/downloads, real model loading, GPU computation, or benchmark execution occurred. Commit/push requested by the user; include the maintained agent context and log, leave the unrelated untracked RL_ROADMAP.md untouched.
