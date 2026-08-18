import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import StandardTransformer, FPSRNModelV1
from fractal_holonet import FractalHoloNet
from benchmark import generate_synthetic_data, train_and_evaluate, benchmark_inference_scaling

def run_experiment():
    torch.manual_seed(42)
    vocab_size = 60
    d_model = 128
    n_layers = 4
    epochs = 6
    
    print("="*75)
    print("      RESEARCH LAB: FRACTAL-HOLONET BENCHMARK & EXPERIMENTS")
    print("="*75)
    
    # 1. Dataset with algorithmic key-value retrieval under distraction
    train_data, _ = generate_synthetic_data(num_samples=2000, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    val_data, _ = generate_synthetic_data(num_samples=500, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    
    models = {
        "1. Standard Transformer (Attention O(N^2))": StandardTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=n_layers, d_ff=512, max_len=4096),
        "2. FPSRN-v1 (Initial Phase Recurrent O(N))": FPSRNModelV1(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=512),
        "3. Fractal-HoloNet (Modernized Holographic Resonance O(N))": FractalHoloNet(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=384)
    }
    
    results = []
    for name, model in models.items():
        res = train_and_evaluate(name, model, train_data, val_data, epochs=epochs, lr=1e-3, batch_size=32)
        results.append(res)
        
    print("\n" + "="*88)
    print(f"{'Model Architecture':<58} | {'Params':<9} | {'Val Loss':<9} | {'Train Time':<10}")
    print("="*88)
    for res in results:
        print(f"{res['model_name']:<58} | {res['params']:<9,} | {res['val_loss']:<9.4f} | {res['train_time']:<8.2f}s")
    print("="*88)
    
    # Latency scaling across sequence context lengths
    lengths = [64, 128, 256, 512, 1024, 2048]
    print("\n--- Long Context Scaling Benchmark (Sequence Lengths 64 -> 2048 tokens) ---")
    latency_results = {}
    for name, model in [
        ("Standard Transformer O(N²)", models["1. Standard Transformer (Attention O(N^2))"]),
        ("Fractal-HoloNet O(N)", models["3. Fractal-HoloNet (Modernized Holographic Resonance O(N))"])
    ]:
        print(f"\nBenchmarking {name}...")
        lats = benchmark_inference_scaling(model, d_model=d_model, seq_lens=lengths)
        latency_results[name] = lats
        
    # Generate clean graph
    plt.figure(figsize=(9, 5.5))
    for name, lats in latency_results.items():
        x_vals = [item[0] for item in lats]
        y_vals = [item[1] for item in lats]
        plt.plot(x_vals, y_vals, marker='o', label=name, linewidth=2.5)
        
    plt.title("Latency Scaling: Transformer Attention O(N²) vs Fractal-HoloNet O(N)", fontsize=12, fontweight='bold')
    plt.xlabel("Context Sequence Length (Tokens)", fontsize=11)
    plt.ylabel("Inference Latency (ms)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join("artifacts", "scaling_comparison.png"), dpi=150)
    print("\n[SUCCESS] Generated visual scaling benchmark report: 'artifacts/scaling_comparison.png'")

if __name__ == "__main__":
    run_experiment()
