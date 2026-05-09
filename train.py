import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import TactIntentGNN
from data_utils import load_match_to_pyg

def pretrain_self_supervised(data_list, device, epochs=20, lr=1e-3):
    model = TactIntentGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(data_list, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
        for data in pbar:
            data = data.to(device)
            N = data.x.size(0)
            if N < 5:
                continue
            
            num_mask = max(1, int(0.3 * N))
            mask_idx = torch.randperm(N)[:num_mask]
            
            node_emb = model(data.x, data.edge_index, data.edge_attr, return_node_emb=True)
            pred_pos = model.node_pred(node_emb[mask_idx])
            true_pos = data.x[mask_idx, :2]
            
            loss = F.mse_loss(pred_pos, true_pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(data_list):.4f}")
    
    return model

def finetune_classifier(model, data_list, device, epochs=15, lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(data_list, desc=f"Finetune Epoch {epoch+1}/{epochs}")
        for data in pbar:
            data = data.to(device)
            if data.y.item() == -1:
                continue
            
            logits, _ = model(data.x, data.edge_index, data.edge_attr)
            loss = F.cross_entropy(logits, data.y.unsqueeze(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == data.y.unsqueeze(0)).sum().item()
            total += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1} loss: {total_loss/max(total,1):.4f} | acc: {acc:.3f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument('--match_ids', nargs='+', default=['3869254', '3869151'])
    parser.add_argument('--epochs_pre', type=int, default=20)
    parser.add_argument('--epochs_ft', type=int, default=15)
    parser.add_argument('--save_dir', default='checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode in ['pretrain', 'both']:
        print("Loading pretraining data...")
        pretrain_data = []
        for mid in args.match_ids:
            pretrain_data.extend(load_match_to_pyg(mid))
        print(f"Loaded {len(pretrain_data)} frames for pretraining")
        
        model = pretrain_self_supervised(pretrain_data, device, epochs=args.epochs_pre)
        torch.save(model.state_dict(), f"{args.save_dir}/gnn_pretrained.pt")
        print("Saved pretrained model")
    else:
        model = TactIntentGNN().to(device)
        model.load_state_dict(torch.load(f"{args.save_dir}/gnn_pretrained.pt", map_location=device))
    
    if args.mode in ['finetune', 'both']:
        print("Loading pseudo-labeled data...")
        import pickle
        with open('data/pseudo_labeled.pkl', 'rb') as f:
            ft_data = pickle.load(f)
        print(f"Loaded {len(ft_data)} labeled frames")
        
        model = finetune_classifier(model, ft_data, device, epochs=args.epochs_ft)
        torch.save(model.state_dict(), f"{args.save_dir}/gnn_final.pt")
        print("Saved final model")
