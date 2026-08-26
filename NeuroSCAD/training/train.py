"""LoRA fine-tuning for prompt → canonical NeuroSCAD IR.

This is an offline job. A checkpoint is never activated in the API automatically;
it must first pass training.evaluate and geometry regression gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SYSTEM = "Convert the engineering request to one canonical NeuroSCAD CSG-IR v0.1 JSON object. Output JSON only."


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model", "train_file", "validation_file", "output_dir"}
    missing = required - config.keys()
    if missing: raise ValueError(f"missing config keys: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("training/config.example.json"))
    args = parser.parse_args(); cfg = load_config(args.config)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install training dependencies: pip install -e '.[training]'") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this production training profile")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], trust_remote_code=False,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    files = {"train": cfg["train_file"], "validation": cfg["validation_file"]}
    dataset = load_dataset("json", data_files=files)
    max_length = int(cfg.get("max_length", 4096))
    system_prompt = str(cfg.get("system_prompt", DEFAULT_SYSTEM))

    def serialize(row: dict) -> tuple[str, str]:
        if tokenizer.chat_template:
            prefix = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prefix = f"System: {system_prompt}\nUser: {row['prompt']}\nAssistant: "
        raw_target = row["target"]
        serialized = raw_target if isinstance(raw_target, str) else json.dumps(raw_target, ensure_ascii=False, separators=(",", ":"))
        return prefix, serialized + tokenizer.eos_token

    # Never train on silently truncated CAD programs: an omitted closing block can
    # teach exactly the failure mode the execution gate is meant to eliminate.
    dataset = dataset.filter(lambda row: len(tokenizer("".join(serialize(row)), add_special_tokens=True)["input_ids"]) <= max_length,
                             desc="Dropping overlength examples")

    def tokenize(row: dict) -> dict:
        prefix, target = serialize(row)
        prefix_ids = tokenizer(prefix, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        encoded = tokenizer(prefix + target, add_special_tokens=True, truncation=True, padding="max_length", max_length=max_length)
        labels = [token if mask else -100 for token, mask in zip(encoded["input_ids"], encoded["attention_mask"])]
        labels[:len(prefix_ids)] = [-100] * min(len(prefix_ids), max_length)
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names, desc="Tokenizing CSG-IR")
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=float(cfg.get("epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(cfg.get("batch_size", 1)), per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation", 16)),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10, eval_strategy="steps", eval_steps=200, save_steps=200,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", seed=int(cfg.get("seed", 42)),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"])
    trainer.train(); trainer.save_model(cfg["output_dir"]); tokenizer.save_pretrained(cfg["output_dir"])
    Path(cfg["output_dir"], "training_manifest.json").write_text(json.dumps({
        "config": cfg, "base_model": cfg["model"],
        "train_examples_after_length_filter": len(tokenized["train"]),
        "validation_examples_after_length_filter": len(tokenized["validation"]),
    }, indent=2) + "\n")

if __name__ == "__main__": main()
