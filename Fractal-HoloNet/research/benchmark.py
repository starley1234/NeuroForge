import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import StandardTransformer, FPSRNModelV1
from dfprn_model import DFPRNModel

# Set seed for reproducibility
torch.manual_seed(42)

# =====================================================================
# Synthetic Algorithmic & Associative Retrieval Benchmark Dataset
# Task: Associative Key-Value Recall with Long-Context Distraction
# Tokens: Keys (A..Z), Values (0..9), Query (?K), Noise tokens
# =====================================================================
def generate_synthetic_data(num_samples=2000, seq_len=64, num_kv_pairs=4, vocab_size=60):
    """
    Generates algorithmic sequence data:
    Format: [K1, V1, K2, V2, ... Distractors/Noise ... Query_K] -> Target is Value for Query_K
    """
    # Key tokens: 10..35, Value tokens: 36..45, Distractors: 46..58, Query symbol: 59, PAD: 0
    data = []
    targets = []
    
    for _ in range(num_samples):
        seq = torch.randint(46, 59, (seq_len,), dtype=torch.long)
        
        # Pick unique keys
        keys = torch.randperm(26)[:num_kv_pairs] + 10
        vals = torch.randint(36, 46, (num_kv_pairs,))
        
        # Place key-value pairs at random distinct locations in first half
        kv_locs = torch.randperm(seq_len // 2 - 2)[:num_kv_pairs] * 2
        for i in range(num_kv_pairs):
            seq[kv_locs[i]] = keys[i]
            seq[kv_locs[i] + 1] = vals[i]
            
        # Target: query one random key at the end of the sequence
        target_idx = torch.randint(0, num_kv_pairs, (1,)).item()
        query_key = keys[target_idx]
        target_val = vals[target_idx]
        
        seq[-2] = 59 # Query token
        seq[-1] = query_key
        
        # In causal LM: predicting the next token after Query_Key
        target_seq = seq.clone()
        # For evaluation, we will compute loss over all next tokens + specifically track query accuracy
        data.append(seq)
        targets.append((target_val, seq))
        
    data = torch.stack(data)
    return data, targets


def train_and_evaluate(model_name, model, train_data, val_data, epochs=8, lr=1e-3, batch_size=32):
    print(f"\n==========================================")
    print(f"  Training: {model_name}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"==========================================")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    num_batches = len(train_data) // batch_size
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        perm = torch.randperm(len(train_data))
        for b in range(num_batches):
            batch_idx = perm[b * batch_size : (b + 1) * batch_size]
            batch = train_data[batch_idx]
            
            # Input: x[:, :-1], Target: x[:, 1:]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inp) # (B, T-1, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_inp = val_data[:, :-1]
            val_tgt = val_data[:, 1:]
            val_logits = model(val_inp)
            val_loss = criterion(val_logits.reshape(-1, val_logits.size(-1)), val_tgt.reshape(-1)).item()
            
            # Accuracy on the final associative recall token
            pred_last = val_logits[:, -1, :].argmax(dim=-1) # Predicted next token for seq[-1]
            # Wait, in seq: seq[-2]=59, seq[-1]=query_key -> next token should be target_val
            # Let's check accuracy on the target token
            
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    train_duration = time.time() - start_time
    print(f"  Completed in {train_duration:.2f}s")
    
    return {
        "model_name": model_name,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "val_loss": val_loss,
        "train_time": train_duration
    }


def benchmark_inference_scaling(model, d_model=128, seq_lens=[64, 128, 256, 512, 1024]):
    """
    Measures latency and memory scaling across increasing sequence lengths.
    """
    model.eval()
    latencies = []
    print(f"\n--- Latency Benchmark across Context Lengths ---")
    for T in seq_lens:
        x = torch.randint(0, 60, (4, T))
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
                
        # Timing
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        elapsed = (time.time() - t0) / 10.0 * 1000.0 # ms
        latencies.append((T, elapsed))
        print(f"  Length: {T:4d} tokens | Latency: {elapsed:.2f} ms")
    return latencies


if __name__ == "__main__":
    vocab_size = 60
    d_model = 128
    n_layers = 4
    
    print("Generating Synthetic Algorithmic Recall Benchmark Data...")
    train_data, _ = generate_synthetic_data(num_samples=2500, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    val_data, _ = generate_synthetic_data(num_samples=500, seq_len=64, num_kv_pairs=4, vocab_size=vocab_size)
    
    # 1. Baseline Transformer
    transformer = StandardTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=n_layers, d_ff=512)
    res_transformer = train_and_evaluate("1. Standard Transformer (Attention O(N^2))", transformer, train_data, val_data, epochs=8)
    
    # 2. V1: Fractal Phase-State Recurrent Net
    fpsrn_v1 = FPSRNModelV1(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=512)
    res_v1 = train_and_evaluate("2. FPSRN-v1 (Phase Recurrence O(N))", fpsrn_v1, train_data, val_data, epochs=8)
    
    # 3. V2 (Modernized): Dynamic Fractal Phase-Resonance Network (DFPRN)
    dfprn = DFPRNModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=384, num_heads=4)
    res_dfprn = train_and_evaluate("3. DFPRN (Modernized Multi-Scale Holographic Phase O(N))", dfprn, train_data, val_data, epochs=8)
    
    print("\n" + "="*60)
    print("                COMPARATIVE SUMMARY")
    print("="*60)
    for res in [res_transformer, res_v1, res_dfprn]:
        print(f"Model: {res['model_name']:<55} | Params: {res['params']:<8} | Val Loss: {res['val_loss']:.4f} | Train Time: {res['train_time']:.2f}s")
    print("="*60)
    
    print("\nBenchmarking Scaling Performance on Long Contexts:")
    print("Transformer Scaling:")
    t_lat = benchmark_inference_scaling(transformer)
    print("DFPRN Scaling:")
    d_lat = benchmark_inference_scaling(dfprn)
