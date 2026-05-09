import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TactIntentGNN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128, out_channels=256,
                 num_classes=12, heads=4, edge_dim=2):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads,
                               concat=True, edge_dim=edge_dim, dropout=0.1)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads,
                               concat=True, edge_dim=edge_dim, dropout=0.1)
        self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads,
                               concat=True, edge_dim=edge_dim, dropout=0.1)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels*heads)
        
        self.node_pred = nn.Linear(hidden_channels*heads, 2)
        
        self.graph_embed = nn.Sequential(
            nn.Linear(hidden_channels*heads, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels)
        )
        self.classifier = nn.Linear(out_channels, num_classes)
    
    def forward(self, x, edge_index, edge_attr, batch=None,
                return_embedding=False, return_node_emb=False):
        x1 = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index, edge_attr))) + x1
        x3 = F.elu(self.bn3(self.conv3(x2, edge_index, edge_attr))) + x2
        
        if return_node_emb:
            return x3
        
        if batch is None:
            x_pool = x3.mean(dim=0, keepdim=True)
        else:
            x_pool = global_mean_pool(x3, batch)
        
        embedding = self.graph_embed(x_pool)
        
        if return_embedding:
            return embedding
        
        logits = self.classifier(embedding)
        return logits, embedding
