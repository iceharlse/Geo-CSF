import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Replay Buffer（经验池），用于存储和采样经验数据
    """
    
    def __init__(self, capacity=100000):
        """
        初始化ReplayBuffer
        
        Args:
            capacity (int): 经验池的最大容量，默认为100000
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, h_graph, preference_set, hv_reward, next_h_graph, done):
        """
        将经验数据存入经验池
        
        Args:
            h_graph (torch.Tensor): 当前状态，形状 [B, condition_dim]
            preference_set (torch.Tensor): 动作（偏好集），形状 [B, N, M]
            hv_reward (torch.Tensor): HV奖励，形状 [B, 1] 或 [B]
            next_h_graph (torch.Tensor): 下一状态，形状 [B, condition_dim]
            done (bool): 是否结束，形状 [B] 或标量
        """
        # 获取批次大小 B
        B = h_graph.shape[0]
            
        # 遍历批次中的每一条经验
        for i in range(B):
            # 提取第 i 条经验 (s, a, r, s', d)
            # 有的张量需要转移到 CPU 以节省 GPU 内存，比如h_graph
            s_i = h_graph[i].cpu()
            a_i = preference_set[i].cpu()
            r_i = hv_reward[i].cpu()
            ns_i = next_h_graph[i].cpu()
            d_i = done[i].cpu()
            
            # 将 *单个* 经验元组存入缓冲区
            experience = (s_i, a_i, r_i, ns_i, d_i)
            self.buffer.append(experience)  
        
    def sample(self, batch_size):
        """
        从经验池中随机采样一批经验数据
        
        Args:
            batch_size (int): 采样批次大小
            
        Returns:
            tuple: 包含5个张量的元组 (s, a, r, next_s, d)
                - s (torch.Tensor): 状态批次，形状 [batch_size, condition_dim]
                - a (torch.Tensor): 动作批次，形状 [batch_size, N, M]
                - r (torch.Tensor): 奖励批次，形状 [batch_size, 1] 或 [batch_size]
                - next_s (torch.Tensor): 下一状态批次，形状 [batch_size, condition_dim]
                - d (torch.Tensor): 完成标志批次，形状 [batch_size]
        """
        # 确保缓冲区中有足够的样本
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer. Requested: {batch_size}, Available: {len(self.buffer)}")
        
        saved_device = torch.get_default_device()
        torch.set_default_device('cpu')
        
            
        # 随机采样batch_size组经验
        samples = random.sample(self.buffer, batch_size)
        
        # 解包采样的经验数据
        h_graph, preference_set, hv_reward, next_h_graph, done = zip(*samples)
        
        # 将列表转换为张量
        s = torch.stack(h_graph, dim=0)
        a = torch.stack(preference_set, dim=0)
        r = torch.stack(hv_reward, dim=0)
        next_s = torch.stack(next_h_graph, dim=0)
        d = torch.stack(done, dim=0).to(dtype=torch.bool)
            
        torch.set_default_device(saved_device)
        
        return s, a, r, next_s, d
        
    def __len__(self):
        """
        返回经验池中存储的经验数量
        
        Returns:
            int: 经验池中存储的经验数量
        """
        return len(self.buffer)
        
    def is_ready(self, batch_size):
        """
        检查经验池是否已准备好进行采样
        
        Args:
            batch_size (int): 需要的批次大小
            
        Returns:
            bool: 如果经验池中的经验数量大于等于batch_size则返回True，否则返回False
        """
        return len(self.buffer) >= batch_size