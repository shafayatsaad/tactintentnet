import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import json
import torch
import numpy as np
from tqdm import tqdm
from model import TactIntentGNN
from data_utils import load_match_to_pyg
from inference import TacticalFingerprintGMM, OTDSCalculator
from agent import generate_alert

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MATCH_ID = 3869685
MODEL_PATH = 'checkpoints/gnn_final.pt'

print("Loading model...")
model = TactIntentGNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Loading match data...")
data_list = load_match_to_pyg(MATCH_ID)

print("Computing embeddings...")
embeddings = []
intent_probs = []
with torch.no_grad():
    for d in tqdm(data_list):
        d = d.to(DEVICE)
        logits, emb = model(d.x, d.edge_index, d.edge_attr)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        embeddings.append(emb.cpu().numpy().flatten())
        intent_probs.append(probs.tolist())

print("Fitting GMM fingerprint...")
gmm = TacticalFingerprintGMM()
gmm.fit(embeddings[:50])

print("Computing OTDS...")
otds_calc = OTDSCalculator(gmm)
otds_timeline = []
alerts = []
for i, emb in enumerate(tqdm(embeddings)):
    minute = (i / len(embeddings)) * 90
    score = otds_calc.update(emb, minute)
    spike, _ = otds_calc.detect_spike(0.6)
    otds_timeline.append({"minute": minute, "otds": score, "spike": spike})
    if spike and (i % 10 == 0):
        alert = generate_alert(score, minute, "2-2", "structural shift")
        alerts.append({"minute": minute, "alert": alert, "otds": score})

print("Computing counterfactuals...")
cf_frames = [len(data_list)//4, len(data_list)//2, 3*len(data_list)//4]
counterfactuals = {}
for fi in cf_frames:
    d = data_list[fi]
    cf_data = []
    for pid in range(min(5, d.x.size(0))):
        for nx in [20, 60, 100]:
            for ny in [20, 40, 60]:
                x_new = d.x.clone()
                x_new[pid, 0] = nx
                x_new[pid, 1] = ny
                
                N = x_new.size(0)
                edge_index = []
                edge_attr = []
                for i in range(N):
                    for j in range(N):
                        if i == j: continue
                        dx = x_new[j, 0] - x_new[i, 0]
                        dy = x_new[j, 1] - x_new[i, 1]
                        dist = np.sqrt(dx.cpu()**2 + dy.cpu()**2)
                        angle = np.arctan2(dy.cpu().item(), dx.cpu().item())
                        edge_index.append([i, j])
                        edge_attr.append([dist, angle])
                
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=DEVICE).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float, device=DEVICE)
                
                with torch.no_grad():
                    logits, _ = model(x_new.to(DEVICE), edge_index, edge_attr)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten().tolist()
                
                cf_data.append({
                    "frame": fi, "player": pid, "x": nx, "y": ny,
                    "probs": probs
                })
    counterfactuals[str(fi)] = cf_data

cache = {
    "match_id": MATCH_ID,
    "n_frames": len(data_list),
    "embeddings": [e.tolist() for e in embeddings],
    "intent_probs": intent_probs,
    "otds_timeline": otds_timeline,
    "alerts": alerts,
    "counterfactuals": counterfactuals,
    "player_positions": [
        {"x": d.x[:,0].tolist(), "y": d.x[:,1].tolist(), "team": d.x[:,4].tolist()}
        for d in data_list
    ]
}

with open('data/demo_cache.json', 'w') as f:
    json.dump(cache, f, default=lambda x: bool(x) if isinstance(x, (np.bool_, np.bool)) else (float(x) if isinstance(x, np.floating) else (int(x) if isinstance(x, np.integer) else x)))

print("Saved demo_cache.json")
