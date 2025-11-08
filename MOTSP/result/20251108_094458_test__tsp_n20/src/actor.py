import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint  # 使用odeint_adjoint替代odeint
from gfp_module import GeometricFeaturePredictor
from csf_module import SetTransformerVectorField


class ODEFuncWrapper(nn.Module):
    """
    包装ODE函数的辅助类，用于torchdiffeq.odeint求解器
    """
    def __init__(self, vector_field, h_graph, g_s):
        super(ODEFuncWrapper, self).__init__()
        self.vector_field = vector_field
        self.h_graph = h_graph
        self.g_s = g_s

    def forward(self, t, Lambda_t):
        """
        ODE函数的前向传播
        
        Args:
            t: 时间参数（标量）
            Lambda_t: 当前时刻的偏好集状态，形状 (B, N, M)
            
        Returns:
            v: 速度场，形状 (B, N, M)
        """
        # vector_field需要t为(B,)形状，而odeint传入的t是标量
        # 我们需要扩展t以匹配批次大小
        batch_size = Lambda_t.shape[0]
        t_batch = t * torch.ones(batch_size, device=Lambda_t.device)
        
        # 计算速度场，传递所有必需的参数
        # vector_field(Lambda_t, t_batch, h_graph, g_s) 返回速度场 v
        v = self.vector_field(Lambda_t, t_batch, self.h_graph, self.g_s)
        return v


class MocoPolicyNetwork(nn.Module):
    """
    Reparameterized Actor-Critic框架的Actor模块
    """
    
    def __init__(self, gfp_params, csf_params, N, M, condition_dim=128):
        """
        初始化MocoPolicyNetwork
        
        Args:
            gfp_params (dict): GFP模块的参数
            csf_params (dict): CSF模块的参数
            N (int): 偏好集大小
            M (int): 目标数
            condition_dim (int): h_graph的维度，默认为128
        """
        super(MocoPolicyNetwork, self).__init__()
        
        self.N = N  # 偏好集大小
        self.M = M  # 目标数
        self.condition_dim = condition_dim
        
        # 实例化Actor部分
        self.gfp = GeometricFeaturePredictor(**gfp_params)
        self.vector_field = SetTransformerVectorField(**csf_params)
        
    def forward(self, h_graph):
        """
        前向传播函数，生成动作
        
        Args:
            h_graph (torch.Tensor): 状态张量，形状 [B, condition_dim]
            
        Returns:
            action (torch.Tensor): 生成的偏好集，形状 [B, N, M]
        """
        B = h_graph.shape[0]  # 批次大小
        
        # Actor部分：生成动作
        # [GFP] 获取几何特征
        g_s = self.gfp(h_graph)
        
        # [ODE] 采样初始噪声 Lambda_0
        # 从标准正态分布中采样（随机性来源）
        # 使用h_graph.device确保设备一致性
        Lambda_0 = torch.randn(B, self.N, self.M, device=h_graph.device)
        
        # [ODE] 定义时间跨度
        # 使用h_graph.device确保设备一致性
        t_span = torch.tensor([0.0, 1.0], device=h_graph.device)
        
        # [ODE] 创建ODEFuncWrapper
        ode_func = ODEFuncWrapper(self.vector_field, h_graph, g_s)
        
        # [ODE] 调用odeint_adjoint求解器（可微分路径，节省内存）
        Lambda_solution = odeint(ode_func, Lambda_0, t_span, method='rk4')
        # 形状 [T, B, N, M], T 是时间点的数量,Lambda_solution[0] 对应 t=0, Lambda_solution[1] 对应 t=1。的偏好集状态
        
        # [Action] 提取最终动作
        raw_action = Lambda_solution[-1]
        # 对动作进行归一化，确保和为 1
        action = torch.softmax(raw_action, dim=-1)
        
        return action