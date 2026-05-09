import json
import os
import numpy as np
import torch
from torch_geometric.data import Data

def load_match_to_pyg(match_id, data_root='~/open-data/data'):
    data_root = os.path.expanduser(data_root)
    path_360 = f'{data_root}/three-sixty/{match_id}.json'
    with open(path_360) as f:
        frames = json.load(f)
    
    data_list = []
    for frame in frames:
        freeze = frame.get('freeze_frame', [])
        if not freeze or len(freeze) < 3:
            continue
        
        nodes = []
        for p in freeze:
            if 'location' not in p:
                continue
            x, y = p['location']
            team = 1.0 if p.get('teammate', False) else 0.0
            nodes.append([x, y, 0.0, 0.0, team])
        
        if len(nodes) < 3:
            continue
        
        x = torch.tensor(nodes, dtype=torch.float)
        N = x.size(0)
        
        edge_index = []
        edge_attr = []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dx = x[j, 0] - x[i, 0]
                dy = x[j, 1] - x[i, 1]
                dist = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy.item(), dx.item())
                edge_index.append([i, j])
                edge_attr.append([dist, angle])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=torch.tensor(-1, dtype=torch.long),
            match_id=match_id, event_id=frame.get('event_uuid', '')
        )
        data_list.append(data)
    
    return data_list

def load_events_df(match_id, data_root='~/open-data/data'):
    import pandas as pd
    data_root = os.path.expanduser(data_root)
    with open(f'{data_root}/events/{match_id}.json') as f:
        events = json.load(f)
    return pd.json_normalize(events)
