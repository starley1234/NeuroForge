#!/usr/bin/env python
"""Мост между дообученной LoRA-моделью и инструментами NEXUS.

Читает промпт со stdin, печатает код OpenSCAD в stdout — ровно то, что ждут
`nexus bench --model "command:..."` и `nexus collect --teacher command`.
Так дообученная модель сравнивается с большой LLM и со своей архитектурой
по одним и тем же метрикам: компилируется ли, manifold ли, держит ли нагрузку.

    echo "<task>Кронштейн под 350 Н" | python scripts/serve_lora.py --adapter artifacts/lora/adapter
"""
from __future__ import annotations

import argparse
import sys

SYSTEM = ("Ты инженер-конструктор. По техническому заданию пишешь параметрический "
          "код OpenSCAD. Единое замкнутое тело, минимальная стенка 1.2 мм, параметры "
          "в начале файла. Отвечай только кодом в блоке ```openscad.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Инференс LoRA-модели для NEXUS")
    ap.add_argument("--adapter", required=True, help="каталог с LoRA-адаптером")
    ap.add_argument("--base", default=None, help="базовая модель, если не в конфиге")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--prompt", default=None, help="вместо чтения stdin")
    args = ap.parse_args()

    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        print("пустой промпт", file=sys.stderr)
        return 2

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = args.base or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if args.four_bit else None

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=args.temperature > 0,
                             temperature=max(args.temperature, 1e-5), top_p=0.95)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    print(answer)          # ```openscad ... ``` — nexus сам достанет код из блока
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
