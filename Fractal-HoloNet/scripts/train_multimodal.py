import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from fractal_holonet.multimodal import MultimodalFractalHoloNet, MultimodalSignalConfig

# =====================================================================
# Генерация реалистичных непрерывных мультимодальных сигналов
# (ЭКГ, Аудио гармоники, Промышленная IoT телеметрия)
# =====================================================================
def generate_synthetic_continuous_signals(num_samples=1000, seq_len=128, channels=1):
    """
    Генерирует квазипериодические непрерывные сигналы с аномалиями и дрейфом фазы (ЭКГ/IoT).
    """
    t = torch.linspace(0, 8 * math.pi, seq_len)
    signals = []
    anomaly_labels = []
    
    for _ in range(num_samples):
        # Базовая гармоника + субгармоники
        f1 = 1.0 + torch.randn(1).item() * 0.1
        f2 = 3.0 + torch.randn(1).item() * 0.2
        phase_shift = torch.rand(1).item() * 2 * math.pi
        
        # ЭКГ-подобная волна (P-QRS-T)
        sig = torch.sin(f1 * t + phase_shift) + 0.5 * torch.sin(f2 * t) + 0.2 * torch.randn(seq_len)
        
        # QRS-подобные пики
        qrs_pulse = torch.exp(-((torch.sin(f1 * t * 0.5 + phase_shift) - 0.8) ** 2) / 0.02) * 2.5
        sig = sig + qrs_pulse
        
        # Добавление искусственной аномалии в 30% случаев
        anom = torch.zeros(seq_len, 1)
        if torch.rand(1).item() < 0.3:
            anom_idx = torch.randint(seq_len // 2, seq_len - 10, (1,)).item()
            sig[anom_idx:anom_idx + 8] += 4.0 # Спайк/аномалия
            anom[anom_idx:anom_idx + 8] = 1.0
            
        signals.append(sig.unsqueeze(-1)) # (seq_len, 1)
        anomaly_labels.append(anom)
        
    return torch.stack(signals), torch.stack(anomaly_labels)


def train_and_verify_multimodal():
    print("=================================================================")
    print("  🚀 MULTIMODAL FRACTAL-HOLONET: CONTINUOUS SIGNAL BENCHMARK")
    print("=================================================================")
    
    seq_len = 128
    channels = 1
    config = MultimodalSignalConfig(
        input_signal_dim=channels,
        output_signal_dim=channels,
        patch_size=1,
        d_model=128,
        n_layers=4,
        d_ff=384,
        use_learnable_fourier_filter=True,
        num_fourier_filters=32,
        vocab_size=0
    )
    
    model = MultimodalFractalHoloNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    
    # 1. Генерация данных
    print("Генерация непрерывных биомедицинских/IoT сигналов (ЭКГ/Аудио)...")
    train_signals, train_anom = generate_synthetic_continuous_signals(num_samples=1200, seq_len=seq_len)
    val_signals, val_anom = generate_synthetic_continuous_signals(num_samples=300, seq_len=seq_len)
    
    # 2. Обучение непрерывной авторегрессии (Forecasting + Anomaly Detection)
    batch_size = 32
    epochs = 8
    num_batches = len(train_signals) // batch_size
    
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    
    print(f"Обучение модели ({sum(p.numel() for p in model.parameters()):,} параметров)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_rec_loss = 0.0
        total_anom_loss = 0.0
        
        perm = torch.randperm(len(train_signals))
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_sig = train_signals[idx]
            batch_anom = train_anom[idx]
            
            # Вход: сигнал t..T-1, Цель: следующий шаг сигнала t+1..T
            inp_sig = batch_sig[:, :-1, :]
            tgt_sig = batch_sig[:, 1:, :]
            tgt_anom = batch_anom[:, 1:, :]
            
            optimizer.zero_grad()
            pred_sig, pred_anom, _ = model.forward_continuous(inp_sig)
            
            loss_rec = mse_criterion(pred_sig, tgt_sig)
            loss_anom = bce_criterion(pred_anom, tgt_anom)
            total_loss = loss_rec + 0.5 * loss_anom
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_rec_loss += loss_rec.item()
            total_anom_loss += loss_anom.item()
            
        avg_rec = total_rec_loss / num_batches
        avg_anom = total_anom_loss / num_batches
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            print(f"  Эпоха [{epoch+1:02d}/{epochs:02d}] | MSE Прогноза: {avg_rec:.5f} | BCE Аномалий: {avg_anom:.5f}")
            
    print(f"✅ Обучение непрерывных сигналов завершено за {time.time() - start_time:.2f} сек!")
    
    # 3. Тест прогнозирования непрерывного потока (O(1) Streaming Forecast)
    print("\n--- 📈 Тестирование Real-Time Streaming Прогнозирования O(1) ---")
    model.eval()
    test_sample = val_signals[:1] # (1, seq_len, 1)
    history = test_sample[:, :64, :]
    actual_future = test_sample[:, 64:, :]
    
    t0 = time.time()
    forecast = model.forecast_stream(history, forecast_steps=64)
    forecast_latency = (time.time() - t0) * 1000.0
    print(f"Прогнозирование 64 шагов вперед выполнено за {forecast_latency:.2f} ms (O(1) шаговая сложность)")
    
    # 4. Визуализация результата
    plt.figure(figsize=(10, 5))
    t_hist = list(range(64))
    t_fore = list(range(64, 128))
    
    plt.plot(t_hist, history[0, :, 0].cpu().numpy(), label="История сигнала (Вход)", color="blue", linewidth=2)
    plt.plot(t_fore, actual_future[0, :, 0].cpu().numpy(), label="Реальный сигнал (Ground Truth)", color="green", linestyle="--", alpha=0.7)
    plt.plot(t_fore, forecast[0, :, 0].cpu().numpy(), label="Прогноз Fractal-HoloNet (O(1) Stream)", color="red", linewidth=2)
    
    plt.title("Непрерывная мультимодальная динамика: Прогнозирование сырого сигнала без токенизации", fontsize=12)
    plt.xlabel("Временные отсчеты (Timesteps)", fontsize=10)
    plt.ylabel("Амплитуда сигнала", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(ROOT / "artifacts" / "multimodal_signal_forecast.png"), dpi=150)
    print("💾 График сохранен в 'artifacts/multimodal_signal_forecast.png'")
    
    # 5. Сохранение чекпоинта мультимодальной модели
    save_dir = str(ROOT / "checkpoints" / "fractal_holonet_multimodal")
    model.save_pretrained(save_dir)
    print(f"💾 Мультимодальный чекпоинт сохранен в: {save_dir}")

if __name__ == "__main__":
    train_and_verify_multimodal()
