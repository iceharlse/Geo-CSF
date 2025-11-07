import torch
import torch.nn as nn

class GeometricFeaturePredictor(nn.Module):
    """
    几何特征预测器(GFP)模块
    从图级别嵌入h_graph中提取低维几何特征表示g_s
    
    输入: h_graph (B, D_h)
    输出: g_s (B, D_g)
    """
    
    def __init__(self, input_dim=128, hidden_dims=[64, 32], output_dim=16, activation='gelu'):
        """
        初始化GFP模块
        
        Args:
            input_dim (int): 输入维度 D_h (默认128，与h_graph维度一致)
            hidden_dims (list): 隐藏层维度列表
            output_dim (int): 输出维度 D_g (默认16)
            activation (str): 激活函数类型 ('relu' 或 'gelu')
        """
        super(GeometricFeaturePredictor, self).__init__()
        
        # 设置激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 添加层归一化提高训练稳定性
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))  # 添加轻微的dropout防止过拟合
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 创建Sequential模型
        self.mlp = nn.Sequential(*layers)
        
        # 保存维度信息
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, h_graph):
        """
        前向传播
        
        Args:
            h_graph (torch.Tensor): 图级别嵌入，形状 (B, D_h)
            
        Returns:
            g_s (torch.Tensor): 几何特征表示，形状 (B, D_g)
        """
        # 通过MLP处理输入
        g_s = self.mlp(h_graph)
        return g_s

def test_geometric_feature_predictor():
    """
    测试GeometricFeaturePredictor模块
    """
    # 创建GFP实例
    gfp = GeometricFeaturePredictor(input_dim=128, hidden_dims=[64, 32], output_dim=16)
    
    # 创建测试输入
    batch_size = 4
    h_graph = torch.randn(batch_size, 128)
    
    # 前向传播
    g_s = gfp(h_graph)
    
    print(f"Input shape: {h_graph.shape}")
    print(f"Output shape: {g_s.shape}")
    print(f"Expected output shape: ({batch_size}, 16)")
    
    # 验证输出形状
    assert g_s.shape == (batch_size, 16), f"Output shape mismatch: {g_s.shape} vs (4, 16)"
    print("Test passed!")

if __name__ == "__main__":
    test_geometric_feature_predictor()