from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel_1layer(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=1, hidden_dim=64):
        super(GNNModel_1layer, self).__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)   # 输入节点特征维度   eg.  x,y,w,h,class_id
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)   # 输入边特征维度     eg. distance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        x = F.relu(x)
        edge_feat = self.edge_encoder(edge_attr)
        edge_feat = F.relu(edge_feat)
        # 图卷积
        x = self.conv1(x, edge_index)
        # 获取边的起点和终点特征
        start, end = edge_index
        start_feat = x[start]
        end_feat = x[end]

        # 边的特征整合
        edge_input = torch.cat([start_feat, end_feat, edge_feat], dim=1)
        out = self.edge_fc(edge_input)
        return out.view(-1)

class GNNModel_2layer(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=1, hidden_dim=64):
        super(GNNModel_2layer, self).__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)   # 输入节点特征维度   eg.  x,y,w,h,class_id
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)   # 输入边特征维度     eg. distance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        x = F.relu(x)
        edge_feat = self.edge_encoder(edge_attr)
        edge_feat = F.relu(edge_feat)
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # 获取边的起点和终点特征
        start, end = edge_index
        start_feat = x[start]
        end_feat = x[end]

        # 边的特征整合
        edge_input = torch.cat([start_feat, end_feat, edge_feat], dim=1)
        out = self.edge_fc(edge_input)
        return out.view(-1)

class GNNModel_3layer(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=1, hidden_dim=64):
        super(GNNModel_3layer, self).__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)   # 输入节点特征维度   eg.  x,y,w,h,class_id
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)   # 输入边特征维度     eg. distance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        x = F.relu(x)
        edge_feat = self.edge_encoder(edge_attr)
        edge_feat = F.relu(edge_feat)
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # 获取边的起点和终点特征
        start, end = edge_index
        start_feat = x[start]
        end_feat = x[end]

        # 边的特征整合
        edge_input = torch.cat([start_feat, end_feat, edge_feat], dim=1)
        out = self.edge_fc(edge_input)
        return out.view(-1)

class GNNModel_4layer(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=1, hidden_dim=64):
        super(GNNModel_4layer, self).__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)   # 输入节点特征维度   eg.  x,y,w,h,class_id
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)   # 输入边特征维度     eg. distance
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.edge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        x = F.relu(x)
        edge_feat = self.edge_encoder(edge_attr)
        edge_feat = F.relu(edge_feat)
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        # 获取边的起点和终点特征
        start, end = edge_index
        start_feat = x[start]
        end_feat = x[end]

        # 边的特征整合
        edge_input = torch.cat([start_feat, end_feat, edge_feat], dim=1)
        out = self.edge_fc(edge_input)
        return out.view(-1)

