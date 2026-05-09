import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import numpy as np
import pickle
from sklearn.cluster import KMeans
from tqdm import tqdm
from model import TactIntentGNN
from data_utils import load_match_to_pyg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TactIntentGNN().to(device)
model.load_state_dict(torch.load('checkpoints/gnn_pretrained.pt', map_location=device))
model.eval()

all_data = []
for mid in ['3869254', '3869151', '3869685']:
    all_data.extend(load_match_to_pyg(mid))

print(f"Encoding {len(all_data)} frames...")
embeddings = []
with torch.no_grad():
    for d in tqdm(all_data):
        d = d.to(device)
        emb = model(d.x, d.edge_index, d.edge_attr, return_embedding=True)
        embeddings.append(emb.cpu().numpy().flatten())

X = np.stack(embeddings, axis=0)
print(f"Embedding shape: {X.shape}")

kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

for i, d in enumerate(all_data):
    d.y = torch.tensor(labels[i], dtype=torch.long)

with open('data/pseudo_labeled.pkl', 'wb') as f:
    pickle.dump(all_data, f)

print("Saved pseudo_labeled.pkl with 12 clusters")
