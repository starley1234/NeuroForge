import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import StandardTransformer, FPSRNModelV1
from dfprn_model import DFPRNModel
from flash_dfprn_model import FlashDFPRNModel
from benchmark import generate_synthetic_data, train_and_evaluate, benchmark_inference_scaling

def run_comprehensive_evaluation():
    torch.manual_seed(42)
    vocab_size = 60
    d_model = 128
    n_layers = 4
    epochs = 8
    
    print("================================================================")
    print("      RESEARCH & BENCHMARK: NOVEL AI ARCHITECTURES")
    print("================================================================")
    
    train_data, _ = generate_synthetic_data(num_samples=3000, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    val_data, _ = generate_synthetic_data(num_samples=600, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    
    models_dict = {
        "Standard Transformer (O(N^2))": StandardTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=n_layers, d_ff=512),
        "FPSRN-v1 (Initial Phase Recurrent)": FPSRNModelV1(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=512),
        "DFPRN-v2 (Multi-Scale Phase)": DFPRNModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=384, num_heads=4),
        "Flash-DFPRN-v3 (Optimized Holographic Resonance)": FlashDFPRNModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=384, num_heads=4)
    }
    
    results = []
    for name, model in models_dict.items():
        res = train_and_evaluate(name, model, train_data, val_data, epochs=epochs, lr=1e-3, batch_size=32)
        results.append(res)
        
    print("\n" + "="*85)
    print(f"{'Model Architecture':<48} | {'Params':<9} | {'Val Loss':<9} | {'Train Time (s)':<12}")
    print("="*85)
    for res in results:
        print(f"{res['model_name']:<48} | {res['params']:<9,} | {res['val_loss']:<9.4f} | {res['train_time']:<12.2f}")
    print("="*85)
    
    # Latency Scaling Test across context lengths
    lengths = [64, 128, 256, 512, 1024, 2048]
    print("\n--- Long Context Scaling Benchmark (Sequence Lengths up to 2048 tokens) ---")
    latency_results = {}
    for name, model in models_dict.items():
        if "v1" in name or "v2" in name:
            continue # Compare Transformer vs Flash-DFPRN v3
        print(f"\nBenchmarking {name}...")
        lats = benchmark_inference_scaling(model, d_model=d_model, seq_lens=lengths)
        latency_results[name] = lats
        
    # Generate visualization plot
    plt.figure(figsize=(10, 6))
    for name, lats in latency_results.items():
        x_vals = [item[0] for item in lats]
        y_vals = [item[1] for item in lats]
        plt.plot(x_vals, y_vals, marker='o', label=name, linewidth=2)
        
    plt.title("Latency Scaling vs Sequence Context Length: Transformer O(N²) vs Flash-DFPRN O(N)", fontsize=13)
    plt.xlabel("Context Length (Tokens)", fontsize=11)
    plt.ylabel("Inference Latency (ms)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join("artifacts", "scaling_comparison.png"), dpi=150)
    print("\nSaved scaling comparison graph to 'artifacts/scaling_comparison.png'")

if __name__ == "__main__":
    run_comprehensive_evaluation()
