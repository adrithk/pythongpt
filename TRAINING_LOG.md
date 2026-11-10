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
| S1 → R1 → S2 → R2 | Planned experiment sequence | [Current plan and comparisons](README.md#current-experiment-plan--november-10-2026) |

A completed PC SFT run and RL results have not been verified.

## Update history

### November 9, 2026 — Documentation started

**User request:** Use this chat to collect information about project training and its results, and maintain a persistent written record.

**Status:** Awaiting the first training update. No new training results reported.

## Details to capture when available

For each run, record its identifier and purpose, date, starting model/checkpoint, dataset and split, hardware and software environment, training settings, duration and cost, metrics and evaluations, saved artifact locations, errors, conclusions, and agreed next steps. Updates can be informal; only supplied or verified details should be filled in.

### November 10, 2026 — Experiment sequence clarified; earlier check verified

The user plans initial SFT on their computer before tiny simple RL, followed by significantly larger/improved SFT and later comprehensive RL after further research. Compare initial versus improved SFT, initial SFT + simple RL versus improved SFT versus improved SFT + comprehensive RL, and each RL checkpoint against its own SFT parent. See README for measurement controls and open recipe choices.

Verified locally today: the saved initial check weights exist; its results record runtime **61.5814 seconds**, train loss **0.6273176551**, and epoch **0.0006844978**. Run date remains unspecified here; the source chats reviewed on November 10 report five updates on Mac MPS. These are training-process observations, not measured coding gains. The intended next local SFT and subsequent RL stages remain planned. No final optimized recipe has been selected.
