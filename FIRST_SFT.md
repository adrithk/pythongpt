# First SFT run

Goal: understand supervised fine-tuning and produce a saved Python coding checkpoint from `Qwen/Qwen2.5-Coder-0.5B`. No required score. This is the SFT portion of v0; RL comes afterward.

This file is a walkthrough for implementation. Training scripts, dependency locks, and prepared datasets do not exist yet, and the AMD setup has not been validated.

## 1. Use the Mac for development and WSL for training

Write all scripts and configurations on the Mac. Prepare and inspect the small dataset there. Transfer the source, configuration, and prepared data to the PC's WSL Linux filesystem. Run training there and bring the result summaries back to the Mac.

Keep the same Python minor version and shared package versions where supported. Install PyTorch separately for each platform: the Mac and ROCm need different builds. Recreate environments on each machine; do not copy a virtual environment between them.

Git can synchronize source and configuration. Transfer prepared data separately with a manifest/checksum; download base model weights directly on the training PC. Keep datasets, caches, environments, and checkpoints outside Git. Start with manual transfer or a private repository if convenient; remote execution can be added later.

AMD documents WSL GPU workloads and supported Ubuntu distributions in its [WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html). Validate the driver/runtime/framework combination on the actual PC before locking it. Check generation, backward propagation, optimizer updates, and checkpoint reload, not just GPU detection.

## 2. Keep the first implementation small

The planned pieces are:

| Piece | Job | Where it runs |
| --- | --- | --- |
| Data preparation | Select examples, deduplicate, split, format, and save a manifest | Mac |
| Training script | Load the base, train, log, and save the model/tokenizer | WSL |
| Generation script | Run fixed prompts with the base or saved checkpoint | WSL initially; optionally Mac later |
| One configuration | Record data paths, model revision, seed, and training settings | Written on Mac; used on WSL |

Use PyTorch, Transformers, Datasets, and TRL's SFTTrainer. Avoid building a general training framework for this run. TRL supports prompt-completion datasets with loss on the completion tokens. [SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer).

## 3. Prepare about 1,000 examples

Use a small Python subset from OpenCoder's `educational_instruct` configuration. The card exposes instructions, code, entry points, and test cases. [Dataset](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2).

Target approximately 1,000 training, 100 development, and 100 test problems after filtering. Remove duplicates and keep related rewrites in the same split. Select short standard-library functions, inspect a sample manually, and verify reference answers against their tests using isolated execution before calling them verified. Retain source IDs and tests separately from model inputs.

Each training example contains a user request and a correct assistant answer. For example:

```text
User: Write a Python function square(n) that returns n multiplied by itself.
Assistant:
def square(n):
    return n * n
```

Use a consistent Qwen chat format and code-only answers. Keep test answers and grader tests out of prompts. Exclude examples that exceed the initial token limit rather than cutting off their solutions. Save the exact examples and split IDs used so later runs can reproduce them.

## 4. Understand what the trainer is doing

During SFT, the model receives the problem and correct solution. At each answer position it predicts the next token. The loss measures how much probability it assigned to the correct token; backpropagation updates the model to make those solution tokens more likely.

Before launching, inspect one tokenized example. Check that the prompt tokens are excluded from the loss, answer tokens are included, and the answer ends with the correct stopping token. This makes the training objective concrete and catches a common implementation error.

## 5. Run a small end-to-end check, then the first SFT job

On WSL, first run a handful of optimizer updates on 16–32 examples. Confirm finite loss, actual parameter updates, memory use, checkpoint save/reload, and successful generation. This checks the whole path before the main job.

Provisional settings for the main run:

| Setting | Starting choice |
| --- | --- |
| Model | Qwen2.5-Coder-0.5B base |
| Data | About 1,000 training examples |
| Epochs | 1 |
| Total sequence limit | 1,024 tokens, keeping only complete examples that fit |
| Microbatch | 1 |
| Gradient accumulation | 16 |
| Full-parameter learning rate | 2e-5 |
| Precision | Mixed precision validated on the actual WSL/ROCm installation |
| Memory | Gradient checkpointing |
| Loss | Completion tokens only |

At these sizes, one epoch is approximately 63 optimizer updates. That is a deliberately small learning experiment, not a promise of a particular quality gain or runtime. Log loss and validation loss, wall time, peak memory, and the exact configuration. Keep checkpoint selection simple and record which checkpoint is used for the comparison.

Try full-parameter training first if the pilot fits in VRAM. If memory is insufficient, reduce sequence length with matching data filtering or use an explicitly configured LoRA fallback. Record the changed recipe; an adapter checkpoint must retain its base model revision to be reloadable. The exact memory fit and compatible software versions remain to be measured.

## 6. Compare before and after, and save the result

Use the same 20–50 held-out prompts for a quick before/after comparison. Save the base answers before training, then generate answers from the saved SFT checkpoint with the same task information and decoding budget. A completion-style base prompt and a chat-style SFT prompt may differ in wrapper; preserve and record that distinction.

Read examples on the Mac and score correctness with the held-out tests in the isolated runner. Track format compliance as well as correctness: a model that follows the requested answer format better has changed, but that alone does not establish better programming ability. Use the development set for debugging; preserve the test set for the final comparison.

Save the checkpoint and tokenizer, software versions, configuration, dataset manifest, training logs, baseline answers, and SFT answers. Reload the saved checkpoint in a fresh process and generate one more answer.

The first SFT run is complete when training and reloading work, you can explain the data and loss, and the result is recorded for future comparisons. A low score or regression is something to inspect, not a reason to discard the experiment. Next, use this checkpoint as the starting point for the small RL run in the main roadmap.
