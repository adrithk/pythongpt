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


@torch.inference_mode()
def generate(tokenizer, model, instruction, max_new_tokens=512):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    inputs = inputs.to(model.device)
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    if any(t is None or t == tokenizer.unk_token_id for t in stop_ids):
        raise ValueError("Expected Qwen stop tokens were not found.")
    settings = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        eos_token_id=stop_ids, pad_token_id=stop_ids[0], use_cache=True,
    )
    output = model.generate(**inputs, generation_config=settings)
    new_ids = output[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    hit_limit = len(new_ids) == max_new_tokens and new_ids[-1].item() not in stop_ids
    return text, bool(hit_limit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    tokenizer, model = load_model(args.model)
    log = Path("runs/inference")
    log.mkdir(parents=True, exist_ok=True)
    log = log / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl")
    print("Answers are saved to", log)
    print("Enter one problem per line, or type /quit. Each prompt starts a fresh conversation.")
    while True:
        if args.prompt_file:
            instruction = args.prompt_file.read_text(encoding="utf-8")
        else:
            try:
                instruction = input("\nProblem> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if instruction == "/quit":
                break
            if not instruction:
                continue
        answer, hit_limit = generate(tokenizer, model, instruction)
        print("\n" + answer)
        if hit_limit:
            print("[Reached the output-token limit; this answer may be incomplete.]")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model": args.model, "prompt": instruction,
                                "answer": answer, "hit_limit": hit_limit}) + "\n")
        if args.prompt_file:
            break


if __name__ == "__main__":
    main()
