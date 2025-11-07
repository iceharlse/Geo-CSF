import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from logging import getLogger
from tqdm import tqdm

from MOTSP.actor import MocoPolicyNetwork
from MOTSP.critic import CriticNetwork
from MOTSP.replay_buffer import ReplayBuffer
from utils.utils import * # (用于 AverageMeter, TimeEstimator, LogData)


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


class GeoCSFTrainer:
    """
    Geo-CSF 训练器
    负责实现 Actor-Critic (DDPG风格) 的训练循环，并管理课程学习
    """
    
    def __init__(self,
                 env, # <--- 接收 moco_env 实例
                 actor_params,
                 critic_params,
                 optimizer_params,
                 trainer_params):
        
        # --- 参数和组件保存 ---
        self.env = env
        self.actor_params = actor_params
        self.critic_params = critic_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # --- 日志和 CUDA (与模板类似) ---
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.device = trainer_params['device']

        # --- 核心组件---
        # 这些组件是 None，它们将在 _initialize_stage 中被创建
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.buffer = None
        self.train_step_counter = 0
        # --- 工具 ---
        self.time_estimator = TimeEstimator()
        
        # --- 获取固定参数 ---
        self.batch_size = self.trainer_params['batch_size']
        self.tau = self.trainer_params['tau']
        self.gamma = self.trainer_params['gamma']
        self.noise_level = self.trainer_params['noise_level']


    def _initialize_stage(self, last_checkpoint_path=None):
        """
        last_checkpoint_path (str or None): 上一阶段的模型路径
        """

        # 1. 初始化 Actor, Critic
        self.actor = MocoPolicyNetwork(**self.actor_params).to(self.device)
        self.critic = CriticNetwork(**self.critic_params).to(self.device)
        
        self.train_step_counter = 0
        
        # 2.热启动
        if last_checkpoint_path is not None:
            # (加载 actor.load_state_dict 和 critic.load_state_dict)
            self.logger.info(f"从 {last_checkpoint_path} 热启动...")
            checkpoint = torch.load(last_checkpoint_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
        # 3. 初始化目标网络
        #self.actor_target = deepcopy(self.actor).to(self.device)
        #self.critic_target = deepcopy(self.critic).to(self.device)
        
        # 4. 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.optimizer_params['lr_actor'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.optimizer_params['lr_critic'])
        
        # 5. 重新初始化 Replay Buffer
        self.buffer = ReplayBuffer(capacity=self.trainer_params['buffer_capacity'])

    def run(self):
        """
        运行训练过程
        """
        # --- 中间层 ---            
        last_checkpoint_path = None # 从头开始
        
        # (检查是否从保存的检查点恢复)
        model_load = self.trainer_params['model_load']
        if model_load['enable']:
            last_checkpoint_path = f"{model_load['path']}/model_N{N}.pth"
            self.logger.info(f"从 {last_checkpoint_path} 恢复训练")
            
        # 获取训练参数
        num_episodes = self.trainer_params.get('num_episodes', 1000)
        N = self.trainer_params.get('N', 10)  # 默认N=10
        
        self.logger.info(f"--- 开始单阶段训练: N = {N}, Episodes = {num_episodes} ---")
        
        # 1. 初始化组件
        self._initialize_stage(last_checkpoint_path)
        
        # 初始化最佳奖励跟踪变量
        self.best_avg_reward = -float('inf')
        
        # 保存初始学习率
        initial_actor_lr = self.optimizer_params['lr_actor']
        initial_critic_lr = self.optimizer_params['lr_critic']
        
        # 2. 执行训练
        self.time_estimator.reset(count=1)
        pbar = tqdm(range(1, num_episodes + 1), desc=f"训练进度 (N={N})")
        
        # 初始化用于记录的变量
        recent_rewards = []
        recent_actor_losses = []
        recent_critic_losses = []
        
        for episode in pbar:
            # A. 收集数据
            step_reward = self._collect_one_step()
            recent_rewards.append(step_reward)
            
            # B. 训练 (如果 buffer 准备好了)
            start_steps = self.trainer_params.get('start_train_after_episodes', 1) * self.batch_size
            
            if self.buffer.is_ready(self.batch_size) and len(self.buffer) >= start_steps:
                actor_loss, critic_loss = self._train_one_batch()
                recent_actor_losses.append(actor_loss) 
                recent_critic_losses.append(critic_loss)
            else:
                # 如果buffer还没准备好，添加占位符
                recent_actor_losses.append(0.0)
                recent_critic_losses.append(0.0)
            
            # 学习率退火 (可选，但强烈推荐)
            # 当 episode 接近 num_episodes 时，new_lr 趋近于 0
            progress = episode / num_episodes
            new_actor_lr = initial_actor_lr * (1.0 - progress)
            new_critic_lr = initial_critic_lr * (1.0 - progress)
            
            # 更新优化器的学习率
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = new_actor_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = new_critic_lr
            
            # 日志记录
            if episode % 100 == 0 or episode == num_episodes:
                # 计算最近100个episode的平均值
                if len(recent_rewards) > 0:
                    avg_reward = np.mean(recent_rewards)
                    avg_actor_loss = 2.0 * np.mean(recent_actor_losses)
                    avg_critic_loss = np.mean(recent_critic_losses)
                    
                    # 更新进度条描述
                    pbar.set_description(f"训练进度 (N={N}, R={avg_reward:.4f}, A_L={avg_actor_loss:.4f}, C_L={avg_critic_loss:.4f})")
                    
                    # 记录到日志
                    self.logger.info(f"Episode {episode}: 平均奖励={avg_reward:.4f}, "
                                   f"Actor损失={avg_actor_loss:.4f}, Critic损失={avg_critic_loss:.4f}")
                    
                    # 早期停止：检查是否是最佳模型
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        # 保存最佳模型
                        self._save_checkpoint(0, N, suffix="_best")
                        self.logger.info(f"Episode {episode}: 保存了新的最佳模型，平均奖励={avg_reward:.4f}")
                
                # 清空记录列表
                recent_rewards.clear()
                recent_actor_losses.clear()
                recent_critic_losses.clear()
            
        # 3. 保存最终检查点
        checkpoint_path = self._save_checkpoint(0, N)  # stage设为0
        
        self.logger.info(" *** 单阶段训练完成！ *** ")


    def _collect_one_step(self):
        """
        与环境交互一步，并将经验存入 buffer
        """
        # --- 输入 ---
        # (self.env, self.actor)
        
        # --- 中间层 ---
        # 1. 重置环境
        state = self.env.reset()
        h_graph = state.h_graph
        
        # 2. Actor 生成动作
        with torch.no_grad():
            action = self.actor(h_graph)
            
        # 3. 添加探索噪声
        # action_noisy = action + torch.randn_like(action) * self.noise_level
        # action_noisy = torch.softmax(action_noisy, dim=-1) # 保证加起来为1
        
        # 4. 与环境交互
        next_state, reward, done, _ = self.env.step(action)
        next_h_graph = next_state.h_graph
        
        # 5. 存储经验
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.tensor(done, dtype=torch.bool)
        
        # (buffer.push 会自动将 GPU 张量移到 CPU 存储)
        self.buffer.push(h_graph, action, reward_tensor, next_h_graph, done_tensor)
        
        # --- 输出 ---
        return np.mean(reward) # (返回标量奖励以便日志记录)


    def _train_one_batch(self):
        """
        从 buffer 中采样并执行一次 Actor-Critic 更新
        """
        # --- 输入 ---
        # (self.buffer, self.actor, self.critic, ...)
        
        self.train_step_counter += 1
        
        # --- 中间层 ---
        # 从 Replay Buffer 采样 (返回 CPU 张量)
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        
        # 将数据移到 GPU
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        ns = ns.to(self.device)
        d = d.to(self.device)
        
        # 训练 Critic
        with torch.no_grad():
        #    a_next = self.actor_target(ns)
        #    q_next = self.critic_target(ns, a_next)
        #    y = r + (1 - d.float().unsqueeze(-1)) * self.gamma * q_next
            y = r
        q_current = self.critic(s, a)
        critic_loss = F.mse_loss(q_current, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss_val = 0.0
        
        # 训练 Actor
        if self.train_step_counter % 2 == 0:
            for p in self.critic.parameters():
                p.requires_grad = False
                
            a_new = self.actor(s)
            actor_loss = -self.critic(s, a_new).mean()
            actor_loss_val = actor_loss.item() # 保存
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for p in self.critic.parameters():
                p.requires_grad = True
            
        # 软更新
        # soft_update(self.actor_target, self.actor, self.tau)
        # soft_update(self.critic_target, self.critic, self.tau)
        
        # --- 输出 ---
        return actor_loss_val, critic_loss.item()
        

    def _save_checkpoint(self, stage, N, suffix=""):
        """
        保存模型检查点
        """
        if suffix == "_best":
            self.logger.info(f"--- 保存最佳模型 (N={N}) ---")
        else:
            self.logger.info(f"--- 训练完成 (N={N}) ---")
        current_checkpoint_path = f"{self.result_folder}/model_N{N}{suffix}.pth"
        
        torch.save({
            'stage': stage,
            'N': N,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }, current_checkpoint_path)
        
        self.logger.info(f"模型已保存到: {current_checkpoint_path}")
        return current_checkpoint_path