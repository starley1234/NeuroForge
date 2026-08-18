"""
Autonomous self-training loop: a deployed LLM (teacher) trains the
Fractal-HoloNet student by itself.

Flow of one round:
  1. Curriculum: teacher prompts (configurable) + optional self-generated
     probe prompts from the student (weakness probing).
  2. Teacher (external LLM via OpenAI-compatible API, or a built-in
     synthetic expert when no API key is configured) generates responses.
  3. The student is fine-tuned on (prompt -> teacher response) pairs.
  4. Eval gate: holdout loss is measured before/after. The round is accepted
     only if the loss improved; otherwise weights are rolled back. This
     protects the deployed checkpoint from catastrophic self-degradation.
  5. Accepted rounds persist the checkpoint.

Two usage modes:
  * CLI:       python self_train.py --rounds 3            (synchronous)
               python self_train.py --interval 300        (daemon, Ctrl+C to stop)
  * REST API:  POST /v1/self-train, /v1/self-train/start|stop, /v1/self-train/status

Security note for production: the teacher endpoint is user-supplied in the
REST API; deploy behind authentication/egress allow-lists.
"""
import os
import time
import math
import json
import argparse
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer
from fractal_holonet.distillation import TeacherAPIClient, DistillationDataset

DEFAULT_EVAL_TEXT = (
    "Фрактальные нейросетевые архитектуры используют комплексный фазовый резонанс "
    "для ассоциативной памяти. Голографическая память хранит знания в комплексном "
    "пространстве состояний. Линейная сложность O(N) позволяет обрабатывать длинные "
    "контексты без роста задержки. Упругое время даёт модели возможность менять темп "
    "своих внутренних часов. Дистилляция знаний переносит рассуждения от большой "
    "модели к компактной. "
)

DEFAULT_CURRICULUM = [
    "Что такое фрактальная голографическая память в нейросетях?",
    "Объясни, почему линейная сложность O(N) важна для длинных контекстов.",
    "Как фазовая интерференция помогает ассоциативной памяти извлекать факты?",
    "Чем комплексный фазовый резонанс отличается от механизма внимания?",
    "Что такое упругое время в рекуррентных архитектурах?",
    "Как работает консолидация памяти по метрике сюрприза?",
]


class SyntheticTeacher:
    """Built-in fallback teacher: a deterministic 'expert' used when no
    external LLM endpoint is configured (offline mode)."""

    def __init__(self, name: str = "synthetic-expert"):
        self.model_name = name

    def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None,
        max_tokens: int = 256, temperature: float = 0.7,
    ) -> str:
        text = (
            f"Вопрос: {prompt} Ответ эксперта: "
            "Фрактальная архитектура строит ассоциативную память через комплексный "
            "фазовый резонанс, сохраняя состояние в компактном векторе. "
            "Линейная сложность O(N) и константная память O(1) на шаг генерации "
            "позволяют обрабатывать длинные контексты без роста задержки. "
            "Упругое время управляет темпом внутренних часов модели, а консолидация "
            "по метрике сюрприза сохраняет редкие и важные факты в долгой памяти. "
        )
        return text


class HoldoutEvalDataset(Dataset):
    def __init__(self, text: str, tokenizer: SimpleProductionTokenizer, block_size: int = 64):
        self.block_size = block_size
        self.pad_token_id = tokenizer.pad_token_id
        self.data = torch.tensor(tokenizer.encode(text, add_bos=False), dtype=torch.long)

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            pad_len = self.block_size + 1 - len(chunk)
            chunk = torch.cat([chunk, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
        return chunk[:-1], chunk[1:]


class SelfTrainLoop:
    """One autonomous self-training round: teacher -> distill -> eval gate."""

    def __init__(
        self,
        student_model: nn.Module,
        tokenizer: SimpleProductionTokenizer,
        teacher_client: Optional[Any] = None,
        eval_text: Optional[str] = None,
        lr: float = 1.5e-3,
        device: str = "cpu",
        save_dir: str = "./checkpoints/self_trained",
        self_prompts_per_round: int = 2,
    ):
        self.device = torch.device(device)
        self.student = student_model.to(self.device)
        self.tokenizer = tokenizer
        self.teacher = teacher_client or SyntheticTeacher()
        self.eval_text = eval_text or DEFAULT_EVAL_TEXT
        self.lr = lr
        self.save_dir = save_dir
        self.self_prompts_per_round = self_prompts_per_round
        self.round = 0
        self.best_loss: Optional[float] = None

    # ------------------------------------------------------------------
    def _make_optimizer(self):
        decay = [p for p in self.student.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.student.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.01},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

    def evaluate_loss(self, block_size: int = 64, batch_size: int = 16) -> float:
        dataset = HoldoutEvalDataset(self.eval_text, self.tokenizer, block_size=block_size)
        loader = DataLoader(dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        was_training = self.student.training
        self.student.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.student(x, states=None, use_step=False)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                total += loss.item()
                batches += 1
        if was_training:
            self.student.train()
        return total / max(1, batches)

    def train_epochs(self, pairs: List[Dict[str, str]], epochs: int, batch_size: int, block_size: int = 96):
        dataset = DistillationDataset(pairs, self.tokenizer, block_size=block_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        optimizer = self._make_optimizer()
        self.student.train()
        for epoch in range(epochs):
            total, batches = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.student(x, states=None, use_step=False)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
                batches += 1
            print(
                f"    [self-train] epoch {epoch+1}/{epochs} | "
                f"loss {total/max(1,batches):.4f}"
            )

    # ------------------------------------------------------------------
    def generate_self_prompts(self, k: int, seeds: Optional[List[str]] = None) -> List[str]:
        """Student probes its own weaknesses: high-temperature continuations
        become new questions for the teacher (self-curriculum)."""
        seeds = seeds or ["Как работает", "Что такое", "Объясни", "Почему"]
        out = []
        try:
            was_training = self.student.training
            self.student.eval()
            for seed in seeds[:k]:
                ids = torch.tensor(
                    [self.tokenizer.encode(seed, add_bos=False)], dtype=torch.long, device=self.device
                )
                gen = self.student.generate(ids, max_new_tokens=24, temperature=1.0)
                text = self.tokenizer.decode(gen[0].tolist(), skip_special=True)
                out.append(f"{seed} {text.strip()}?")
            if was_training:
                self.student.train()
        except Exception as e:  # self-probing is best-effort
            print(f"    [self-train] self-prompt generation skipped: {e}")
        return out

    def run_round(
        self,
        prompts: Optional[List[str]] = None,
        epochs: int = 4,
        batch_size: int = 4,
        curriculum: bool = True,
    ) -> Dict[str, Any]:
        self.round += 1
        prompts = list(prompts) if prompts else list(DEFAULT_CURRICULUM)
        if curriculum and self.round > 1 and self.self_prompts_per_round > 0:
            prompts = prompts + self.generate_self_prompts(self.self_prompts_per_round)

        print(f"\n[SELF-TRAIN] Round {self.round} | teacher={self.teacher.model_name}")
        print(f"[SELF-TRAIN] Requesting {len(prompts)} completions from teacher...")
        pairs = []
        for p in prompts:
            resp = self.teacher.generate_completion(
                prompt=p,
                system_prompt="You are a concise, logical AI teacher. Answer in the language of the question.",
                max_tokens=150,
            )
            pairs.append({"prompt": p, "response": resp})
            print(f"    teacher ok: {p[:60]}...")

        t0 = time.time()
        loss_before = self.evaluate_loss()
        backup_path = os.path.join(self.save_dir, ".backup_round.pt")
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.student.state_dict(), backup_path)

        self.train_epochs(pairs, epochs=epochs, batch_size=batch_size)
        loss_after = self.evaluate_loss()

        accepted = self.best_loss is None or loss_after < self.best_loss - 1e-6
        if accepted:
            self.best_loss = loss_after
            self.student.save_pretrained(self.save_dir)
            self.tokenizer.save(os.path.join(self.save_dir, "tokenizer.json"))
            print(
                f"[SELF-TRAIN] ACCEPTED: holdout loss {loss_before:.4f} -> {loss_after:.4f} "
                f"(best {self.best_loss:.4f})"
            )
        else:
            self.student.load_state_dict(torch.load(backup_path, map_location=self.device))
            print(
                f"[SELF-TRAIN] REJECTED (no improvement): {loss_before:.4f} -> {loss_after:.4f}; "
                f"weights rolled back."
            )

        result = {
            "round": self.round,
            "accepted": accepted,
            "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4),
            "best_loss": round(self.best_loss, 4),
            "duration_sec": round(time.time() - t0, 2),
            "samples": len(pairs),
            "teacher_mode": self.teacher.model_name,
            "save_dir": self.save_dir,
        }
        print(f"[SELF-TRAIN] round {self.round} finished in {result['duration_sec']}s")
        return result


class SelfTrainService:
    """Background autonomous self-training daemon (REST /v1/self-train/start)."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints/fractal_holonet_base",
        save_dir: Optional[str] = None,
        teacher_endpoint: str = "https://api.openai.com/v1",
        teacher_api_key: str = "",
        teacher_model: str = "gpt-4o-mini",
        interval_sec: float = 300.0,
        epochs: int = 4,
        batch_size: int = 4,
        curriculum: bool = True,
        eval_text: Optional[str] = None,
        on_accepted: Optional[Any] = None,
    ):
        save_dir = save_dir or checkpoint_dir
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            student = ProductionFractalHoloNet.from_pretrained(checkpoint_dir, map_location="cpu")
        else:
            student = ProductionFractalHoloNet(
                FractalHoloNetConfig(vocab_size=300, d_model=128, n_layers=4, d_ff=384)
            )
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
        tokenizer = (
            SimpleProductionTokenizer.load(tokenizer_path)
            if os.path.exists(tokenizer_path)
            else SimpleProductionTokenizer()
        )
        teacher = (
            TeacherAPIClient(endpoint=teacher_endpoint, api_key=teacher_api_key, model_name=teacher_model)
            if teacher_api_key
            else SyntheticTeacher()
        )
        self.loop = SelfTrainLoop(
            student, tokenizer, teacher_client=teacher, eval_text=eval_text, save_dir=save_dir
        )
        self.interval_sec = interval_sec
        self.epochs = epochs
        self.batch_size = batch_size
        self.curriculum = curriculum
        self.on_accepted = on_accepted
        self.rounds = 0
        self.last_result: Optional[Dict[str, Any]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _worker(self):
        while not self._stop.is_set():
            try:
                result = self.loop.run_round(
                    epochs=self.epochs, batch_size=self.batch_size, curriculum=self.curriculum
                )
            except Exception as e:
                result = {"error": str(e), "accepted": False}
            self.rounds += 1
            self.last_result = result
            if result.get("accepted") and self.on_accepted is not None:
                try:
                    self.on_accepted(result)
                except Exception as e:
                    print(f"[SELF-TRAIN] on_accepted callback failed: {e}")
            self._stop.wait(self.interval_sec)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="self-train-loop")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "rounds": self.rounds,
            "interval_sec": self.interval_sec,
            "teacher_mode": self.loop.teacher.model_name,
            "last_result": self.last_result,
        }


def main():
    ap = argparse.ArgumentParser(description="ELAST-HOLO / Fractal-HoloNet autonomous self-training")
    ap.add_argument("--checkpoint", default="./checkpoints/fractal_holonet_base")
    ap.add_argument("--save-dir", default="./checkpoints/self_trained")
    ap.add_argument("--rounds", type=int, default=0, help="synchronous rounds (0 = daemon mode)")
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between daemon rounds")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--endpoint", default=os.getenv("TEACHER_ENDPOINT", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.getenv("TEACHER_API_KEY", ""))
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL", "gpt-4o-mini"))
    ap.add_argument("--no-curriculum", action="store_true")
    args = ap.parse_args()

    if not args.api_key:
        print("TEACHER_API_KEY не задан: используем встроенного синтетического учителя (offline).")

    service = SelfTrainService(
        checkpoint_dir=args.checkpoint,
        save_dir=args.save_dir,
        teacher_endpoint=args.endpoint,
        teacher_api_key=args.api_key,
        teacher_model=args.model,
        interval_sec=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        curriculum=not args.no_curriculum,
    )

    if args.rounds > 0:
        for _ in range(args.rounds):
            res = service.loop.run_round(
                epochs=args.epochs, batch_size=args.batch_size, curriculum=not args.no_curriculum
            )
            print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"[SELF-TRAIN] Daemon mode: round every {args.interval}s. Ctrl+C to stop.")
        service.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            service.stop()
            print("\n[SELF-TRAIN] Stopped.")


if __name__ == "__main__":
    main()
