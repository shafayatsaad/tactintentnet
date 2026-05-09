import numpy as np
import torch
import time
from model import TactIntentGNN
from data_utils import load_match_to_pyg

device = torch.device('cuda')
model = TactIntentGNN().to(device)
model.load_state_dict(torch.load('checkpoints/gnn_final.pt', map_location=device))
model.eval()

# Load one frame
data = load_match_to_pyg(3869685)[0]
data = data.to(device)

# Warmup
with torch.no_grad():
    _ = model(data.x, data.edge_index, data.edge_attr)

# Benchmark
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        logits, emb = model(data.x, data.edge_index, data.edge_attr)
    torch.cuda.synchronize()
    times.append(time.time() - t0)

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"HIP version: {torch.version.hip}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Mean inference time: {np.mean(times)*1000:.2f}ms")
print(f"Throughput: {1/np.mean(times):.1f} frames/sec")
print(f"Embedding shape: {emb.shape}")
print(f"Top intent: {logits.argmax().item()} (confidence: {torch.softmax(logits, -1).max().item():.3f})")
