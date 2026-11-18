# Finished and replaced work

Moved here on November 15, 2026 to keep the main folder to what the next run
uses. Nothing was deleted, and Git history follows each file.

| File | What it was | Replaced by |
| --- | --- | --- |
| `init_prepare.py` | Built the first run's data (`data/sft-splits`) | `curate_*.py` |
| `init_inspect.py` | Checked that data for overlap and length | `curate_build.py` report |
| `FIRST_SFT.md` | Walkthrough for the first run | `CLOUD.md`, `CURATE.md` |
| `deepseek_distill.py`, `DEEPSEEK_DISTILL.md` | First small DeepSeek collector | `distill_*.py`, `DISTILL_STAGES.md` |
| `deepseek_pipeline.py`, `DEEPSEEK_PIPELINE.md` | Full verification and upload pipeline | `distill_verify.py`, `curate_*.py` |
| `DISTILLATION_PLAN.md` | Design notes for that pipeline | `DISTILL_STAGES.md` |
| `tests/` | Offline tests for the two DeepSeek scripts | `tests/` in the main folder |

`init_trainsft.py` is the first run's training script, built on SFTTrainer. It was later edited into the second
recipe for an 80 GB rented GPU (3 epochs, LR 5e-5, warmup, cosine, 1,024
tokens), so it now holds the settings to carry into `train_torch.py`. The exact
first-run version is in Git history. It was replaced by `train_torch.py`.

These scripts were written to sit in the main folder. The two `init_` scripts
use paths relative to where you run them, so run them from the main folder
(`python initial/init_prepare.py`). The two DeepSeek scripts look for `data/`
next to themselves, so move them back before running them again. The tests
still run from here:

~~~bash
cd initial && python -m unittest discover -s tests
~~~
