import torch

def compute_dpp_reward(actions, sigma=0.2, epsilon=1e-6):
    """
    计算行列式点过程 (DPP) 奖励 R_DPP = log(det(K))
    
    Args:
        actions (torch.Tensor): 偏好集, 形状 (B, N, M)
        sigma (float): 高斯核的带宽
        epsilon (float): 用于数值稳定性的对角扰动
        
    Returns:
        torch.Tensor: DPP 奖励, 形状 (B, 1)
    """
    # actions 形状: (B, N, M)
    B, N, M = actions.shape
    
    # 1. 计算成对距离的平方 ||v_i - v_j||^2
    diffs = actions.unsqueeze(2) - actions.unsqueeze(1)
    sq_dists = (diffs ** 2).sum(dim=-1) # (B, N, N)
    
    # 2. 计算高斯核 K_ij = exp(-sq_dists / (2*sigma^2))
    kernel_matrix = torch.exp(-sq_dists / (2 * (sigma**2))) # (B, N, N)
    
    # 3. 添加 epsilon*I 保证数值稳定性 (Jitter)
    identity = torch.eye(N, device=actions.device).unsqueeze(0).expand(B, -1, -1)
    stable_kernel = kernel_matrix + epsilon * identity
    
    # 4. 使用 torch.linalg.slogdet 计算 log(det(K))
    #    slogdet 返回 (sign, logabsdet)，我们只需要 logabsdet
    #    这比直接计算 det(K) 更数值稳定
    _, log_det_K = torch.linalg.slogdet(stable_kernel)
    
    # 返回形状 (B, 1)
    return log_det_K.unsqueeze(-1)