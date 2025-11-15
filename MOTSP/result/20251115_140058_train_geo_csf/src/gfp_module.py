import torch
import torch.nn as nn
import math

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

class GeometricFeaturePredictor(nn.Module):
    """
    几何特征预测器(GFP)模块
    从节点嵌入node_embeddings中提取低维几何特征表示g_s
    
    输入: node_embeddings (B, N, embedding_dim)
    输出: g_s (B, D_g)
    """
    
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=16, num_layers=2, num_heads=8, ff_hidden_dim=512):
        """
        初始化GFP模块
        
        Args:
            input_dim (int): 输入节点嵌入维度 (默认128)
            hidden_dim (int): Set Transformer内部的主要隐藏维度 (默认128)
            output_dim (int): 输出几何特征维度 D_g (默认16)
            num_layers (int): Set Transformer块的数量 (默认2)
            num_heads (int): 多头注意力机制的头数 (默认8)
            ff_hidden_dim (int): FFN内部隐藏层维度 (默认512)
        """
        super(GeometricFeaturePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        
        # 输入投影层: 将输入的节点嵌入 (embedding_dim维) 投影到 hidden_dim 维
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 全局池化层: 将节点级别的表示聚合为图级别的表示
        self.global_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Set Transformer 块列表
        self.layers = nn.ModuleList([
            SetTransformerBlock(hidden_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 输出投影层: 将最终的 hidden_dim 维表示投影到 output_dim 维几何特征
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, node_embeddings):
        """
        前向传播
        
        Args:
            node_embeddings (torch.Tensor): 节点嵌入，形状 (B, N, embedding_dim)
            
        Returns:
            g_s (torch.Tensor): 几何特征表示，形状 (B, D_g)
        """
        batch_size, num_nodes, embedding_dim = node_embeddings.shape
        
        # 输入投影: x = self.input_projection(node_embeddings)
        # 得到 x (形状 (B, N, hidden_dim))
        x = self.input_projection(node_embeddings)
        
        # 生成全局上下文向量 (通过平均池化节点表示)
        global_context = torch.mean(x, dim=1)  # (B, hidden_dim)
        c = self.global_pooling(global_context)  # (B, hidden_dim)
        
        # 通过 Set Transformer 块
        for layer in self.layers:
            x = layer(x, c)
        
        # 聚合节点表示为图级表示 (使用平均池化)
        graph_representation = torch.mean(x, dim=1)  # (B, hidden_dim)
        
        # 输出投影: g_s = self.output_projection(graph_representation)
        # 得到 g_s (形状 (B, output_dim))
        g_s = self.output_projection(graph_representation)
        
        return g_s