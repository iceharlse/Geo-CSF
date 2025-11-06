import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os
import sys
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MOTSP.actor import MocoPolicyNetwork
from MOTSP.critic import CriticNetwork
from MOTSP.replay_buffer import ReplayBuffer
from MOTSP.moco_env import MOCOEnv, State


def soft_update(target, source, tau):
    """
    软更新目标网络参数
    
    Args:
        target (nn.Module): 目标网络
        source (nn.Module): 源网络
        tau (float): 更新率
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def main():
    # 超参数设置
    BATCH_SIZE = 32
    CAPACITY = 100000
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-4
    GAMMA = 0.99
    TAU = 0.005  # 软更新参数
    NOISE_LEVEL = 0.1  # 探索噪声水平
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    M = 2   # 目标数（固定）
    condition_dim = 128  # h_graph的维度
    
    # 环境参数
    model_params = {
        'embedding_dim': 128,
        'encoder_layer_num': 6,
        'head_num': 8,
        'qkv_dim': 16,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'sqrt_embedding_dim': 16,
        'logit_clipping': 10.0
    }
    
    env_params = {
        'problem_size': 20,
        'pomo_size': 20
    }
    
    # 模型参数（不依赖于N，可以放在循环外部）
    gfp_params = {
        'input_dim': 128,  # 与condition_dim一致
        'hidden_dims':[64, 32],
        'output_dim': 16
    }
    
    # 修正：csf_params的input_dim必须等于M（偏好向量的维度）
    csf_params = {
        'input_dim': M,
        'hidden_dim': 128,
        'condition_dim': condition_dim, 
        'geometric_dim': gfp_params['output_dim'],
        'time_embed_dim': 128,
        'num_layers': 2,
        'num_heads': 8,
        'ff_hidden_dim': 512
    }
    
    
    # 定义课程学习阶段
    CURRICULUM_STAGES = [
        (10, 1000),   # 阶段 1: 训练 N=10，跑 1000 个 episodes
        (20, 2000),   # 阶段 2: 训练 N=20，跑 2000 个 episodes
        (50, 5000),   # 阶段 3: 训练 N=50，跑 5000 个 episodes
        (101, 10000)  # 阶段 4: 最终训练 N=101，跑 10000 个 episodes
    ]
    
    LAST_CHECKPOINT_PATH = None  # 用于热启动
    
    # 检查点路径
    checkpoint_path = os.path.join(script_dir, 'POMO', 'result', 'train__tsp_n20', 'checkpoint_motsp-200.pt')
    
    # 初始化环境（不依赖N，可以放在循环外部）
    print("初始化环境...")
    env = MOCOEnv(
        weca_model_params=model_params,
        weca_env_params=env_params,
        weca_checkpoint_path=checkpoint_path,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        ref_point=[20.0, 20.0]
    )
    
    # --- 外层课程循环 ---
    for stage, (N, num_episodes) in enumerate(CURRICULUM_STAGES):
        print(f"--- 开始课程学习阶段 {stage+1}: N = {N}, Episodes = {num_episodes} ---")
        
        # 初始化Actor和Critic网络（依赖N，必须在循环内部）
        print("初始化Actor和Critic网络...")
        actor = MocoPolicyNetwork(gfp_params, csf_params, N, M, condition_dim).to(DEVICE)
        critic = CriticNetwork(condition_dim, N, M).to(DEVICE)
        
        # 热启动逻辑
        if LAST_CHECKPOINT_PATH is not None:
            print(f"  ... 从 {LAST_CHECKPOINT_PATH} 加载权重进行热启动...")
            try:
                checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
                
                # 加载模型权重
                actor.load_state_dict(checkpoint['actor_state_dict'])
                critic.load_state_dict(checkpoint['critic_state_dict'])
                
                # 注意：由于N可能变化，优化器状态可能不兼容，所以我们不加载优化器状态
                print("  ... 成功加载模型权重 (优化器状态因N变化而重新初始化)")
            except Exception as e:
                print(f"  ... 加载检查点失败: {e}，将从头开始训练")
        
        # 创建目标网络
        print("创建目标网络...")
        actor_target = deepcopy(actor).to(DEVICE)
        critic_target = deepcopy(critic).to(DEVICE)
        
        # 冻结目标网络的梯度计算
        for param in actor_target.parameters():
            param.requires_grad = False
        for param in critic_target.parameters():
            param.requires_grad = False
        
        # 初始化优化器（依赖模型参数，必须在循环内部）
        actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
        critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)
        
        # 初始化经验池（必须在循环内部，避免不同N的样本混合）
        buffer = ReplayBuffer(capacity=CAPACITY)
        
        # 训练循环
        print("开始训练...")
        for episode in tqdm(range(num_episodes), desc=f"训练进度 (N={N})"):
            # A. 收集数据 (与环境交互)
            state = env.reset()  # 获取 h_graph
            h_graph = state.h_graph  # 形状 [B, condition_dim]
            
            # 通过Actor生成偏好集
            with torch.no_grad():
                action = actor(h_graph)  # 形状 [B, N, M]
            
            # 添加噪声进行探索
            action_noisy = action + torch.randn_like(action) * NOISE_LEVEL
            # 确保动作仍然在有效的概率分布范围内
            action_noisy = torch.softmax(action_noisy, dim=-1)
            
            # 与环境交互
            next_state, reward, done, info = env.step(action_noisy)
            next_h_graph = next_state.h_graph  # 形状 [B, condition_dim]
            
            # 存储经验到经验池
            # 注意：reward是numpy数组，需要转换为tensor
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # 形状 [B, 1]
            done_tensor = torch.tensor(done, dtype=torch.bool, device=DEVICE)  # 形状 [B]
            
            buffer.push(h_graph, action_noisy, reward_tensor, next_h_graph, done_tensor)
            
            # B. 训练 (从经验池采样)
            if buffer.is_ready(BATCH_SIZE):
                # 从Replay Buffer中获取旧数据
                s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(BATCH_SIZE)
                
                # 将数据移到正确的设备上
                s_batch = s_batch.to(DEVICE)
                a_batch = a_batch.to(DEVICE)
                r_batch = r_batch.to(DEVICE)
                ns_batch = ns_batch.to(DEVICE)
                d_batch = d_batch.to(DEVICE)
                
                # C. 训练Critic (LCritic)
                with torch.no_grad():
                    # 让目标Actor生成下一个动作
                    a_next = actor_target(ns_batch)  # 形状 [B, N, M]
                    
                    # 让目标Critic评估这个未来
                    q_next = critic_target(ns_batch, a_next)  # 形状 [B, 1]
                    
                    # 计算目标Q值
                    y = r_batch + (1 - d_batch.float().unsqueeze(-1)) * GAMMA * q_next  # 形状 [B, 1]
                
                # 让当前Critic评估当时的(s, a)对
                q_current = critic(s_batch, a_batch)  # 形状 [B, 1]
                
                # 计算Critic损失
                critic_loss = F.mse_loss(q_current, y)
                
                # 更新Critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # D. 训练Actor (LActor)
                # 冻结Critic
                for p in critic.parameters():
                    p.requires_grad = False
                
                # 让当前Actor重新为这批s生成新的动作
                a_new = actor(s_batch)  # 形状 [B, N, M]
                
                # Actor损失：希望Critic的打分最高
                actor_loss = -critic(s_batch, a_new).mean()
                
                # 更新Actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # 解冻Critic
                for p in critic.parameters():
                    p.requires_grad = True
                
                # E. 软更新目标网络
                soft_update(actor_target, actor, TAU)
                soft_update(critic_target, critic, TAU)
            
            # 每100个episode打印一次信息
            if episode % 100 == 0:
                print(f"Episode {episode}, Buffer size: {len(buffer)}")
        
        print(f"--- 阶段 {stage+1} (N={N}) 训练完成! ---")
        
        # 保存阶段性检查点
        current_checkpoint_path = f"model_stage_N{N}.pth"
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        }, current_checkpoint_path)
        
        print(f"阶段性模型已保存到: {current_checkpoint_path}")
        
        # 为下一个循环阶段设置"热启动"路径
        LAST_CHECKPOINT_PATH = current_checkpoint_path
    
    print("所有课程学习阶段完成！")


if __name__ == "__main__":
    main()