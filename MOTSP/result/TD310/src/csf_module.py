import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码，用于时间嵌入"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t):
        """
        Args:
            t: 时间张量，形状为 (B,) 或标量
            
        Returns:
            位置编码，形状为 (B, dim)
        """
        # 确保t是(B,)形状
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

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

class SetTransformerVectorField(nn.Module):
    """
    Set Transformer向量场函数 v_θ，用于流匹配模型
    接收当前时刻的偏好集 Λ_t、时间 t 以及条件 h_graph 和 g_s，输出速度场 v = dΛ_t/dt
    
    输入形状:
        Lambda_t: (B, N, M) - 当前偏好集
        t: (B,) 或标量 - 时间
        h_graph: (B, D_h) - 图条件向量
        g_s: (B, D_g) - 几何特征向量
        
    输出形状:
        v: (B, N, M) - 速度场
    """
    
    def __init__(self, **model_params):
        """
        初始化方法
        
        Args:
            model_params (dict): 模型参数字典
                - input_dim (M): 输入/输出偏好向量的维度 (例如: 2)
                - hidden_dim (d_model): Set Transformer 内部的主要隐藏维度 (例如: 128)
                - condition_dim (D_h): 输入的 h_graph 条件向量维度 (例如: 128)
                - geometric_dim (D_g): 输入的 g_s 几何特征向量维度 (例如: 16)
                - num_layers (L): Set Transformer 块的数量 (例如: 2-6 层)
                - num_heads: 多头注意力机制的头数 (例如: 8)
                - ff_hidden_dim (d_ff): FFN 内部隐藏层维度 (通常是 4 * d_model)
                - time_embed_dim (可选): 时间嵌入 e_t 的维度 (可以设为 d_model)
        """
        super().__init__()
        
        # 提取参数
        self.input_dim = model_params.get('input_dim', 2)
        self.hidden_dim = model_params.get('hidden_dim', 128)
        self.condition_dim = model_params.get('condition_dim', 128)
        self.geometric_dim = model_params.get('geometric_dim', 16)  # 新增几何特征维度
        self.num_layers = model_params.get('num_layers', 2)
        self.num_heads = model_params.get('num_heads', 8)
        self.ff_hidden_dim = model_params.get('ff_hidden_dim', 4 * self.hidden_dim)
        self.time_embed_dim = model_params.get('time_embed_dim', self.hidden_dim)
        
        # 输入投影层: 将输入的 Λ_t (M维) 投影到 d_model 维
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 时间编码层: 使用正弦位置编码将标量时间 t 映射到 time_embed_dim 维向量 e_t
        self.time_encoding = SinusoidalPositionalEncoding(self.time_embed_dim)
        
        # 条件融合层: 将 h_graph、g_s 和 e_t 融合成一个 d_model 维的上下文向量 c
        self.condition_fusion = nn.Sequential(
            nn.Linear(self.condition_dim + self.geometric_dim + self.time_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Set Transformer 块列表
        self.layers = nn.ModuleList([
            SetTransformerBlock(self.hidden_dim, self.num_heads, self.ff_hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # 输出投影层: 将最终的 d_model 维表示投影回原始的 M 维速度向量
        self.output_projection = nn.Linear(self.hidden_dim, self.input_dim)
        
    def forward(self, Lambda_t, t, h_graph, g_s):
        """
        主前向传播方法
        
        Args:
            Lambda_t: (B, N, M) - 当前时刻的偏好集
            t: (B,) 或标量 - 时间
            h_graph: (B, D_h) - 图条件向量
            g_s: (B, D_g) - 几何特征向量
            
        Returns:
            v: (B, N, M) - 速度场
        """
        batch_size = Lambda_t.size(0)
        
        # 时间编码: 将 t 转换为 e_t (形状 (B, time_embed_dim))
        e_t = self.time_encoding(t)
        
        # 条件融合: c = self.condition_fusion(torch.cat((h_graph, g_s, e_t), dim=-1))
        # 得到 c (形状 (B, hidden_dim))
        c = self.condition_fusion(torch.cat((h_graph, g_s, e_t), dim=-1))
        
        # 输入投影: x = self.input_projection(Lambda_t)
        # 得到 x (形状 (B, N, hidden_dim))
        x = self.input_projection(Lambda_t)
        
        # 通过 Set Transformer 块
        for layer in self.layers:
            x = layer(x, c)
        
        # 输出投影: v = self.output_projection(x)
        # 得到 v (形状 (B, N, M))
        v = self.output_projection(x)
        
        return v

# 简单的测试代码
if __name__ == "__main__":
    # 测试参数
    model_params = {
        'input_dim': 2,
        'hidden_dim': 128,
        'condition_dim': 128,
        'geometric_dim': 16,  # 新增几何特征维度
        'num_layers': 2,
        'num_heads': 8,
        'ff_hidden_dim': 512,
        'time_embed_dim': 128
    }
    
    # 创建模型实例
    model = SetTransformerVectorField(**model_params)
    
    # 测试输入
    batch_size, seq_len, input_dim = 4, 10, 2
    condition_dim = 128
    geometric_dim = 16
    
    # 使用softmax确保Lambda_t最后一个维度的元素和为1
    Lambda_t_raw = torch.randn(batch_size, seq_len, input_dim)
    Lambda_t = torch.softmax(Lambda_t_raw, dim=-1)
    t = torch.randn(batch_size)
    h_graph = torch.randn(batch_size, condition_dim)
    g_s = torch.randn(batch_size, geometric_dim)  # 新增几何特征输入
    
    # 验证Lambda_t最后一个维度的和为1
    print(f"Lambda_t最后一个维度的和: {Lambda_t.sum(dim=-1)[0, 0]}")  # 应该接近1.0
    
    # 前向传播
    with torch.no_grad():
        v = model(Lambda_t, t, h_graph, g_s)
    
    print(f"输入 Lambda_t 形状: {Lambda_t.shape}")
    print(f"输入 t 形状: {t.shape}")
    print(f"输入 h_graph 形状: {h_graph.shape}")
    print(f"输入 g_s 形状: {g_s.shape}")
    print(f"输出 v 形状: {v.shape}")
    print("测试通过!")