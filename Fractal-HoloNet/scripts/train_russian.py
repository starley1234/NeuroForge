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
from fractal_holonet.datasets import get_russian_corpus, ByteDataset


def train_russian_language():
    print("=" * 75)
    print("  🇷🇺 ОБУЧЕНИЕ МОДЕЛИ FRACTAL-HOLONET РУССКОМУ ЯЗЫКУ (RUSSIAN PRE-TRAINING)")
    print("=" * 75)
    
    corpus = get_russian_corpus()
    encoded_bytes = corpus.encode("utf-8")
    print(f"📊 Размер обучающего корпуса: {len(corpus):,} символов ({len(encoded_bytes):,} UTF-8 байт)")
    
    tokenizer = SimpleProductionTokenizer()
    tokens = tokenizer.encode(corpus, add_bos=False)
    
    block_size = 96
    batch_size = 64
    dataset = ByteDataset(tokens, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"📦 Всего обучающих батчей: {len(loader)} (блок {block_size} токенов, batch_size={batch_size})")
    
    config = FractalHoloNetConfig(
        vocab_size=300,
        d_model=128,
        n_layers=4,
        d_ff=384,
        dropout=0.02
    )
    model = ProductionFractalHoloNet(config)
    print(f"⚙️ Параметров архитектуры: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 75)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    save_dir = str(ROOT / "checkpoints" / "fractal_holonet_base")
    
    start_time = time.time()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        
        for x, y in loader:
            optimizer.zero_grad()
            logits, _ = model(x, states=None, use_step=False)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / max(1, batches)
        ppl = math.exp(min(avg_loss, 20))
        print(f"  Эпоха [{epoch+1:02d}/{epochs:02d}] | Train Loss: {avg_loss:.4f} | Perplexity (PPL): {ppl:.2f}")
            
    total_time = time.time() - start_time
    print(f"✅ Обучение русскому языку завершено за {total_time:.2f} сек!")
    
    # Сохраняем модель
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"💾 Обученная русскоязычная модель сохранена в: {save_dir}")
    
    # Комплексное тестирование русскоязычных ответов
    print("\n" + "=" * 75)
    print("  🧪 ТЕСТИРОВАНИЕ РУССКОЯЗЫЧНЫХ ОТВЕТОВ FRACTAL-HOLONET")
    print("=" * 75)
    
    pipe = FractalHoloNetInferencePipeline(save_dir)
    
    prompts = [
        "Искусственный интеллект — это",
        "Архитектура Fractal-HoloNet построена на",
        "Вопрос: В чем главное преимущество фрактальной архитектуры?\nОтвет:",
        "Пользователь: Какая сложность вычислений при генерации текста?\nАссистент:",
        "Мороз и солнце; день чудесный!"
    ]
    
    for p in prompts:
        res = pipe.generate(p, max_new_tokens=100, temperature=0.5, top_k=25, top_p=0.85)
        print(f"\n[ПРОМПТ]:\n{p}")
        print(f"[ОТВЕТ МОДЕЛИ]:\n{res['generated_text']}")
        print(f"[ПОЛНЫЙ ТЕКСТ]:\n{res['full_text']}")
        print("-" * 50)

if __name__ == "__main__":
    train_russian_language()
