import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


def load_model(model_path):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable. Check your environment and ROCm exports."
        )
    model_path = Path(model_path)

    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No config.json found in {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(
                f"Nonfinite saved weights in {name}. "
                "Preserve this checkpoint for inspection."
            )

    print("Saved weights are finite.")
    print("GPU:", torch.cuda.get_device_name(0))

    model = model.to("cuda")
    model.eval()

    return tokenizer, model
