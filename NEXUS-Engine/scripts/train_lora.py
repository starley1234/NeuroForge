#!/usr/bin/env python
"""LoRA/QLoRA-дообучение современной открытой модели на данных NEXUS.

Зачем: обучать 200M+ с нуля дорого и долго, а дообучить готовую модель-кодер на
своём домене можно за вечер на одной RTX 5060 Ti (16 ГБ).

Что делает скрипт:
  1. читает train.jsonl/val.jsonl в chat-формате (см. `nexus export-sft`);
  2. поднимает базовую модель через Unsloth (если установлен) или PEFT+TRL;
  3. вешает LoRA-адаптеры и учит с параметрами, проверенными на Blackwell;
  4. сохраняет адаптер (и по флагу — слитые веса / GGUF).

Установка (CUDA 12.8 для sm_120):
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install unsloth trl peft transformers datasets accelerate bitsandbytes

Запуск:
    python scripts/train_lora.py --data artifacts/sft --model Qwen/Qwen3-Coder-8B \
        --epochs 3 --batch-size 2 --grad-accum 8 --seq-len 4096

Проверка результата тем же мерилом, что и своей модели:
    nexus bench --model "command:python scripts/serve_lora.py --adapter out/lora" \
                --model "command:<ваша большая LLM>" --prompts real_prompts.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Модели, проверенные на 16 ГБ (август 2026). Лицензии: Apache-2.0 у Qwen и
# Mistral Small, MIT у GLM, Apache-2.0 у gpt-oss.
PRESETS = {
    # 7-9B: LoRA в bf16 помещается целиком, самый предсказуемый вариант
    "qwen3-coder-8b":   dict(model="Qwen/Qwen3-Coder-8B",        load_in_4bit=False),
    "qwen3-8b":         dict(model="Qwen/Qwen3-8B",              load_in_4bit=False),
    "mistral-small-4":  dict(model="mistralai/Mistral-Small-4",  load_in_4bit=True),
    # 13-24B: только QLoRA 4-bit, длину последовательности придётся уменьшить
    "codestral-22b":    dict(model="mistralai/Codestral-22B-v0.1", load_in_4bit=True),
    "devstral-small-2": dict(model="mistralai/Devstral-Small-2",   load_in_4bit=True),
}


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA-дообучение на данных NEXUS")
    ap.add_argument("--data", default="artifacts/sft", help="каталог с train.jsonl/val.jsonl")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="готовый выбор базовой модели")
    ap.add_argument("--model", default=None, help="явное имя модели на HuggingFace")
    ap.add_argument("--out", default="artifacts/lora")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="эффективный батч = bs*accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=None, help="по умолчанию = rank")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--four-bit", action="store_true", help="QLoRA вместо LoRA bf16")
    ap.add_argument("--no-unsloth", action="store_true", help="только PEFT+TRL")
    ap.add_argument("--merge", action="store_true", help="сохранить слитые веса")
    ap.add_argument("--gguf", default=None, help="квант для экспорта, напр. q4_k_m")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = build_args()
    preset = PRESETS[args.preset] if args.preset else {}
    model_name = args.model or preset.get("model") or PRESETS["qwen3-coder-8b"]["model"]
    load_in_4bit = args.four_bit or bool(preset.get("load_in_4bit", False))
    alpha = args.alpha or args.rank

    train_file = os.path.join(args.data, "train.jsonl")
    val_file = os.path.join(args.data, "val.jsonl")
    if not os.path.exists(train_file):
        print(f"нет {train_file}. Сначала: nexus export-sft --source "
              f"jsonl:artifacts/ingest/dataset.jsonl --out {args.data}", file=sys.stderr)
        return 2

    import torch
    from datasets import load_dataset

    print(f"[lora] база: {model_name}, 4-bit: {load_in_4bit}, "
          f"seq_len: {args.seq_len}, rank: {args.rank}")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    use_unsloth = not args.no_unsloth
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("[lora] unsloth не установлен — иду через PEFT (медленнее в ~2 раза)")
            use_unsloth = False

    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=args.seq_len,
            dtype=None,                      # bf16 на Blackwell определится сам
            load_in_4bit=load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout,
            target_modules=target_modules, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                       bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout, bias="none",
            task_type="CAUSAL_LM", target_modules=target_modules))
        model.gradient_checkpointing_enable()

    data_files = {"train": train_file}
    if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
        data_files["validation"] = val_file
    dataset = load_dataset("json", data_files=data_files)

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(to_text, remove_columns=["messages"])
    print(f"[lora] примеров: train={len(dataset['train'])}, "
          f"val={len(dataset.get('validation', []))}")

    from trl import SFTConfig, SFTTrainer
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if "validation" in dataset else "no",
        load_best_model_at_end="validation" in dataset,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        max_seq_length=args.seq_len,
        packing=True,
        dataset_text_field="text",
        seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, args=config,
                         train_dataset=dataset["train"],
                         eval_dataset=dataset.get("validation"))
    trainer.train()

    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[lora] адаптер сохранён: {adapter_dir}")

    if args.merge and use_unsloth:
        merged = os.path.join(args.out, "merged")
        model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
        print(f"[lora] слитые веса: {merged}")
    if args.gguf and use_unsloth:
        model.save_pretrained_gguf(os.path.join(args.out, "gguf"), tokenizer,
                                   quantization_method=args.gguf)
        print(f"[lora] GGUF ({args.gguf}) готов — можно грузить в Ollama/llama.cpp")

    with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": model_name, "four_bit": load_in_4bit,
                   "rank": args.rank, "alpha": alpha, "epochs": args.epochs,
                   "seq_len": args.seq_len, "unsloth": use_unsloth},
                  fh, indent=2, ensure_ascii=False)
    print("[lora] дальше: nexus bench --model \"command:python scripts/serve_lora.py "
          f"--adapter {adapter_dir}\" --prompts real_prompts.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
