import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Critic网络，用于评估(state, action)对的Q值
    """
    
    def __init__(self, condition_dim, N, M, geometric_dim=16, hidden_dim=128):
        """
        初始化CriticNetwork
        
        Args:
            condition_dim (int): 状态h_graph的维度
            N (int): 偏好集大小
            M (int): 目标数
            geometric_dim (int): 几何特征g_s的维度，默认为16
            hidden_dim (int): 隐藏层维度，默认为128
        """
        super(CriticNetwork, self).__init__()
        
        self.condition_dim = condition_dim
        self.geometric_dim = geometric_dim
        self.N = N
        self.M = M
        self.hidden_dim = hidden_dim
        
        # 状态处理网络
        self.state_processor = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 几何特征处理网络
        self.geometric_processor = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作处理网络
        self.action_processor = nn.Sequential(
            nn.Linear(self.N*self.M, hidden_dim),
            nn.ReLU()
        )
        
        # 拼接后的处理网络
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h_graph, g_s, action):
        """
        前向传播函数，计算Q值
        
        Args:
            h_graph (torch.Tensor): 状态张量，形状 [B, condition_dim]
            g_s (torch.Tensor): 几何特征张量，形状 [B, geometric_dim]
            action (torch.Tensor): 动作张量，形状 [B, N, M]
            
        Returns:
            q_value (torch.Tensor): Q值，形状 [B, 1]
        """
        # 处理状态 h_graph
        processed_state = self.state_processor(h_graph)  # [B, hidden_dim]
        
        # 处理几何特征 g_s
        processed_geometric = self.geometric_processor(g_s)  # [B, hidden_dim]
        
        # 动作
        flattened_action = action.view(action.size(0), -1) # [B, N*M]
        processed_action = self.action_processor(flattened_action) # [B, 128]
        
        # 拼接处理后的状态、几何特征和动作
        combined = torch.cat([processed_state, processed_geometric, processed_action], dim=1)  # [B, hidden_dim*3]
        
        # 计算最终的Q值
        q_value = self.combined_processor(combined)  # [B, 1]
        
        return q_value