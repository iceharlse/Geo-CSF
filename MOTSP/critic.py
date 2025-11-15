# critic.py
import torch
import torch.nn as nn


class SetTransformerBlock(nn.Module):
    """Set Transformer块，包含自注意力、条件注意力和前馈网络"""
    def __init__(self, hidden_dim, num_heads, ff_hidden_dim):
        """
        Args:
            hidden_dim: 隐藏维度 d_model
            num_heads: 多头注意力头数
            ff_hidden_dim: 前馈网络隐藏维度 d_ff
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Pre-LN层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 使用PyTorch内置的多头注意力层
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(0.1)
        self.attn_dropout_cab = nn.Dropout(0.1)
        
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, c):
        """
        Args:
            x: 集合表示，形状为 (B, N, d_model)
            c: 条件上下文，形状为 (B, d_model)
            
        Returns:
            更新后的集合表示，形状为 (B, N, d_model)
        """
        # Pre-LN + 自注意力
        norm1_out = self.norm1(x)
        # 使用PyTorch内置的多头注意力
        attn_output, _ = self.self_attn(norm1_out, norm1_out, norm1_out)
        attn_output = self.attn_dropout(attn_output)
        
        # 残差连接
        x = x + attn_output
        
        # Pre-LN + 条件注意力 (交叉注意力)
        norm2_out = self.norm2(x)
        # 扩展条件c以匹配序列长度
        c_expanded = c.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 使用PyTorch内置的交叉注意力
        attn_output_cab, _ = self.cross_attn(norm2_out, c_expanded, c_expanded)
        attn_output_cab = self.attn_dropout_cab(attn_output_cab)
        
        # 残差连接
        x = x + attn_output_cab
        
        # Pre-LN + FFN
        norm3_out = self.norm3(x)
        ffn_out = self.ffn(norm3_out)
        
        # 残差连接
        x = x + ffn_out
        
        return x

class CriticNetwork(nn.Module):
    """
    Critic网络, 使用Set Transformer处理 node_embeddings 和 action 两个集合
    """
    
    def __init__(self, node_embedding_dim, N, M, hidden_dim=128, num_heads=8, ff_hidden_dim=512, num_layers=2, geometric_dim=16):
        """
        初始化CriticNetwork
        
        Args:
            node_embedding_dim (int): 节点嵌入的维度 (例如 128)
            N (int): 偏好集大小
            M (int): 目标数
            hidden_dim (int): 隐藏层维度
            num_heads (int): 多头注意力头数
            ff_hidden_dim (int): 前馈网络隐藏维度
            num_layers (int): Set Transformer块的数量
            geometric_dim (int): 几何特征维度
        """
        super(CriticNetwork, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.N = N
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_layers = num_layers
        self.geometric_dim = geometric_dim

        # --- 1. 问题状态处理器 (处理 node_embeddings) ---
        # 输入投影层: 将输入的节点嵌入投影到 hidden_dim 维
        self.problem_input_projection = nn.Linear(self.node_embedding_dim, hidden_dim)
        
        # Set Transformer 块列表 (用于处理问题状态)
        self.problem_transformer_blocks = nn.ModuleList([
            SetTransformerBlock(hidden_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])

        # --- 2. 动作集处理器 (处理 action) ---
        # 输入投影层: 将动作投影到 hidden_dim 维
        self.action_input_projection = nn.Linear(self.M, hidden_dim)
        
        # Set Transformer 块列表 (用于处理动作集)
        self.action_transformer_blocks = nn.ModuleList([
            SetTransformerBlock(hidden_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # --- 3. 几何特征处理器 (处理 g_s) ---
        self.geometric_processor = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- 4. 拼接后的处理网络 ---
        # 输入维度 = problem_embedding(h) + action_embedding(h) + geometric_embedding(h)
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 只输出HV
        )
        
    def forward(self, node_embeddings, action, g_s):
        """
        前向传播函数
        
        Args:
            node_embeddings (torch.Tensor): 节点嵌入 [B, P, D_emb] (P=problem_size)
            action (torch.Tensor): 动作张量 [B, N, M]
            g_s (torch.Tensor, optional): 几何特征张量 [B, geometric_dim]
            
        Returns:
            q_value (torch.Tensor): Q值 [B, 1]
        """
        
        # 1. 处理问题状态 node_embeddings
        # 输入投影: (B, P, D_emb) -> (B, P, hidden_dim)
        problem_x = self.problem_input_projection(node_embeddings)
        
        # 生成全局上下文向量 (通过平均池化节点表示)
        problem_global_context = torch.mean(problem_x, dim=1)  # (B, hidden_dim)
        
        # 通过 Set Transformer 块处理问题状态
        for block in self.problem_transformer_blocks:
            problem_x = block(problem_x, problem_global_context)
        
        # 聚合节点表示为图级表示 (使用平均池化)
        problem_pooled = torch.mean(problem_x, dim=1)  # (B, hidden_dim)
        
        # 2. 处理动作集 action
        # 输入投影: (B, N, M) -> (B, N, hidden_dim)
        action_x = self.action_input_projection(action)
        
        # 生成全局上下文向量 (通过平均池化动作表示)
        action_global_context = torch.mean(action_x, dim=1)  # (B, hidden_dim)
        
        # 通过 Set Transformer 块处理动作集
        for block in self.action_transformer_blocks:
            action_x = block(action_x, action_global_context)
        
        # 聚合动作表示为集合级表示 (使用平均池化)
        action_pooled = torch.mean(action_x, dim=1)  # (B, hidden_dim)
        
        # 3. 处理几何特征 g_s (如果提供)
        if g_s is not None:
            geometric_pooled = self.geometric_processor(g_s)  # [B, hidden_dim]
            # 4. 拼接处理后的状态、动作和几何特征
            combined = torch.cat([problem_pooled, action_pooled, geometric_pooled], dim=1)  # [B, hidden_dim*3]
        else:
            # 如果没有提供g_s，则只使用problem_pooled和action_pooled
            combined = torch.cat([problem_pooled, action_pooled], dim=1)  # [B, hidden_dim*2]
        
        # 5. 计算最终的Q值
        q_value = self.combined_processor(combined)  # [B, 1]
        
        return q_value