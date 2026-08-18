import os
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline
from fractal_holonet.datasets import ByteDataset as TextDataset

def train_on_benchmark_corpus():
    data_path = str(ROOT / "data" / "data_benchmark.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found! Please download data_benchmark.txt first.")
        
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        
    print("=" * 70)
    print("  🚀 TRAINING FRACTAL-HOLONET ON STANDARD BENCHMARK (TinyShakespeare)")
    print("=" * 70)
    print(f"📊 Общий объем текста: {len(full_text):,} символов ({len(full_text.encode('utf-8')):,} байт)")
    
    # 90% train / 10% validation split
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]
    
    tokenizer = SimpleProductionTokenizer()
    train_tokens = tokenizer.encode(train_text, add_bos=False)
    val_tokens = tokenizer.encode(val_text, add_bos=False)
    
    block_size = 128
    batch_size = 64
    
    train_dataset = TextDataset(train_tokens, block_size=block_size)
    val_dataset = TextDataset(val_tokens, block_size=block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📦 Обучающих батчей: {len(train_loader)} | Валидационных: {len(val_loader)}")
    
    # Инициализируем модель Fractal-HoloNet
    config = FractalHoloNetConfig(
        vocab_size=300,
        d_model=128,
        n_layers=4,
        d_ff=384,
        dropout=0.05
    )
    model = ProductionFractalHoloNet(config)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"⚙️ Параметров модели: {param_count:,}")
    print("-" * 70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 1
    save_dir = str(ROOT / "checkpoints" / "fractal_holonet_base")
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, _ = model(x, states=None, use_step=False)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if (step + 1) % 40 == 0 or (step + 1) == len(train_loader):
                print(f"  [Эпоха {epoch+1}/{epochs}] Шаг {step+1:03d}/{len(train_loader):03d} | Train Loss: {loss.item():.4f}")
                
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                logits_val, _ = model(x_val, states=None, use_step=False)
                v_loss = criterion(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1))
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(min(avg_val_loss, 20))
        print(f"  ⭐ Итог эпохи {epoch+1:02d} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f}")
        print("-" * 70)
        
    duration = time.time() - start_time
    print(f"✅ Обучение успешно завершено за {duration:.2f} сек!")
    
    # Сохраняем обученный чекпоинт
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"💾 Обученная модель сохранена в: {save_dir}")
    
    # Тест генерации обученной модели
    print("\n--- 🎭 Тест генерации Fractal-HoloNet после обучения на Шекспире ---")
    pipe = FractalHoloNetInferencePipeline(save_dir)
    test_prompts = [
        "First Citizen:\nBefore we proceed",
        "KING RICHARD:\nWhat said our cousin",
        "ROMEO:\nLady, by yonder"
    ]
    for prompt in test_prompts:
        res = pipe.generate(prompt, max_new_tokens=80, temperature=0.7)
        print(f"\n[Промпт]:\n{prompt}")
        print(f"[Генерация Fractal-HoloNet]:\n{res['generated_text']}")
        print("~" * 50)

if __name__ == "__main__":
    train_on_benchmark_corpus()
