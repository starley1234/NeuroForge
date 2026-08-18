import os
import time
import math
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline


class TeacherAPIClient:
    """
    OpenAI-совместимый HTTP-клиент для вызова Teacher модели (OpenAI, DeepSeek, OpenRouter, vLLM, Ollama и др.)
    через endpoint, model_name и api_key.
    """
    def __init__(self, endpoint: str, api_key: str, model_name: str, timeout: int = 30):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions") and not self.endpoint.endswith("/completions"):
            self.endpoint = f"{self.endpoint}/chat/completions"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def generate_completion(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.7) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        req_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(self.endpoint, data=req_data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                choices = res_json.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"]
                return ""
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Teacher API Error ({e.code}): {err_msg}")
        except Exception as e:
            raise RuntimeError(f"Teacher Connection Error: {str(e)}")


class DistillationDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer: SimpleProductionTokenizer, block_size: int = 128):
        self.block_size = block_size
        self.samples = []

        pad_id = tokenizer.pad_token_id
        for item in pairs:
            # Формируем цепочку Промпт -> Рассуждение/Ответ Учителя.
            # <eos> — НАСТОЯЩИЙ спец-токен токенизатора, а не литеральная строка.
            prefix = tokenizer.encode(f"User: {item['prompt']}\nAssistant: {item['response']}\n", add_bos=False)
            suffix = tokenizer.encode("\n", add_bos=False)
            token_ids = prefix + [tokenizer.eos_token_id] + suffix

            # Чанкуем на блоки фиксированной длины
            for i in range(0, len(token_ids) - 1, block_size):
                chunk = token_ids[i:i + block_size + 1]
                if len(chunk) < block_size + 1:
                    pad_len = block_size + 1 - len(chunk)
                    chunk = chunk + [pad_id] * pad_len
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            return self.samples[0]
        return self.samples[idx]


class FractalHoloNetDistiller:
    """
    Модуль дистилляции знаний (Knowledge Distillation) из мощной Teacher LLM в компактную Fractal-HoloNet Student.
    Поддерживает:
    1. Автоматическую генерацию синтетического датасета и Chain-of-Thought рассуждений от Teacher.
    2. Обучение Student модели с адаптивной регуляризацией и сходимостью.
    3. Сохранение обновленного чекпоинта студента.
    """
    def __init__(
        self,
        student_model: ProductionFractalHoloNet,
        tokenizer: SimpleProductionTokenizer,
        teacher_client: Optional[TeacherAPIClient] = None,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.student = student_model.to(self.device)
        self.tokenizer = tokenizer
        self.teacher = teacher_client
        
        decay_params = [p for p in self.student.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.student.parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ], lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def distill_from_pairs(
        self,
        dataset_pairs: List[Dict[str, str]],
        epochs: int = 5,
        batch_size: int = 4,
        block_size: int = 128,
        save_dir: str = "./checkpoints/fractal_holonet_base"
    ) -> Dict[str, Any]:
        """
        Дистилляция на парах (Prompt, Teacher Response).
        """
        print(f"\n🚀 Запуск дистилляции знаний в Fractal-HoloNet...")
        print(f"📚 Обучающих примеров от Teacher: {len(dataset_pairs)}")
        print(f"⚙️ Параметров Student-модели: {sum(p.numel() for p in self.student.parameters() if p.requires_grad):,}")
        print("-" * 65)

        dataset = DistillationDataset(dataset_pairs, self.tokenizer, block_size=block_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.student.train()
        start_time = time.time()
        history_loss = []

        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits, _ = self.student(x, states=None, use_step=False)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            avg_loss = total_loss / max(1, batches)
            history_loss.append(avg_loss)
            print(f"  Эпоха [{epoch+1:02d}/{epochs:02d}] | Distillation Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20)):.2f}")

        duration = time.time() - start_time
        print(f"✅ Дистилляция завершена за {duration:.2f} сек!")

        os.makedirs(save_dir, exist_ok=True)
        self.student.save_pretrained(save_dir)
        self.tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
        print(f"💾 Обновленный Student чекпоинт сохранен в: {save_dir}\n")

        return {
            "status": "success",
            "epochs": epochs,
            "final_loss": round(history_loss[-1], 4) if history_loss else 0.0,
            "duration_sec": round(duration, 2),
            "save_dir": save_dir
        }

    def distill_from_teacher_api(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = "You are a concise, helpful and logical AI.",
        epochs: int = 5,
        batch_size: int = 4,
        save_dir: str = "./checkpoints/fractal_holonet_base"
    ) -> Dict[str, Any]:
        """
        Генерирует ответы через Teacher API (OpenAI / OpenRouter / Ollama) и обучает Student модель.
        """
        if self.teacher is None:
            raise ValueError("Teacher API Client не сконфигурирован! Передайте endpoint, model и api_key.")

        print(f"\n📡 Запрос знаний у Teacher модели ({self.teacher.model_name}) по {len(prompts)} промптам...")
        distill_pairs = []
        for i, p in enumerate(prompts, 1):
            print(f"  [Запрос {i}/{len(prompts)}]: '{p}'")
            teacher_resp = self.teacher.generate_completion(prompt=p, system_prompt=system_prompt, max_tokens=150)
            distill_pairs.append({"prompt": p, "response": teacher_resp})

        return self.distill_from_pairs(distill_pairs, epochs=epochs, batch_size=batch_size, save_dir=save_dir)
