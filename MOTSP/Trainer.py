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
        self.critic_1 = None
        self.critic_2 = None
        self.actor_target = None
        self.critic_1_target = None
        self.critic_2_target = None
        self.actor_optimizer = None
        self.critic_1_optimizer = None
        self.critic_2_optimizer = None
        self.buffer = None
        self.train_step_counter = 0
        # --- 工具 ---
        self.time_estimator = TimeEstimator()
        
        # --- 获取固定参数 ---
        self.batch_size = self.trainer_params['batch_size']


    def _initialize_stage(self, last_checkpoint_path=None):
        """
        last_checkpoint_path (str or None): 上一阶段的模型路径
        """

        #  初始化 Actor, Critic
        self.actor = MocoPolicyNetwork(**self.actor_params).to(self.device)
        self.critic_1 = CriticNetwork(**self.critic_params).to(self.device)
        self.critic_2 = CriticNetwork(**self.critic_params).to(self.device)
        
        self.train_step_counter = 0
        
        
        #  初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.optimizer_params['lr_actor'])
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.optimizer_params['lr_critic'])
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.optimizer_params['lr_critic'])
        
        # 重新初始化 Replay Buffer
        self.buffer = ReplayBuffer(capacity=self.trainer_params['buffer_capacity'])
        
        self.avg_reward = AverageMeter()
        self.avg_actor_loss = AverageMeter()
        self.avg_critic_loss_total = AverageMeter() # (这个是总的_加权_损失)
        
        # --- (!! 核心 !!) ---
        # (这是用于监控尺度的_未加权_损失)
        self.avg_unweighted_hv_loss = AverageMeter() 

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
        
        
        for episode in pbar:
            # A. 收集数据
            step_reward = self._collect_one_step()
            self.avg_reward.update(step_reward)
            
            # B. 训练 (如果 buffer 准备好了)
            start_steps = self.trainer_params.get('start_train_after_episodes', 1) * self.batch_size
            
            if self.buffer.is_ready(self.batch_size) and len(self.buffer) >= start_steps:
                
                # --- (修改) 捕获所有返回的损失值 ---
                actor_loss, critic_loss, unweighted_hv = self._train_one_batch()
                
                # --- (新) 更新 Meters ---
                self.avg_actor_loss.update(actor_loss)
                self.avg_critic_loss_total.update(critic_loss)
                self.avg_unweighted_hv_loss.update(unweighted_hv)
                
            else:
                # (如果buffer没准备好，可以 update 0.0)
                self.avg_actor_loss.update(0.0)
                self.avg_critic_loss_total.update(0.0)
                self.avg_unweighted_hv_loss.update(0.0)
            
            # 学习率退火 (可选，但强烈推荐)
            # 当 episode 接近 num_episodes 时，new_lr 趋近于 0
            progress = episode / num_episodes
            new_actor_lr = initial_actor_lr * (1.0 - progress)
            new_critic_lr = initial_critic_lr * (1.0 - progress)
            
            # 更新优化器的学习率
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = new_actor_lr
            for param_group in self.critic_1_optimizer.param_groups:
                param_group['lr'] = new_critic_lr
            for param_group in self.critic_2_optimizer.param_groups:
                param_group['lr'] = new_critic_lr
            
            # 日志记录
            if episode % 100 == 0 or episode == num_episodes:
                
                # --- (修改) 从 Meters 中获取平均值 ---
                if self.avg_reward.count > 0: # (检查是否有数据)
                    avg_reward = self.avg_reward.avg
                    avg_actor_loss = 3.0 * self.avg_actor_loss.avg
                    avg_critic_loss = self.avg_critic_loss_total.avg
                    
                    # --- (!! 核心 !!) ---
                    avg_L_hv_unweighted = self.avg_unweighted_hv_loss.avg
                    # --- (!! 核心结束 !!) ---
                    
                    # 更新进度条描述
                    pbar.set_description(f"训练进度 (N={N}, R={avg_reward:.4f}, A_L={avg_actor_loss:.4f}, C_L={avg_critic_loss:.4f})")
                    
                    # --- (!! 核心 !!) ---
                    # 记录到日志 (这里是你监控尺度的地方)
                    self.logger.info(f"Episode {episode}: 平均奖励={avg_reward:.4f}")
                    self.logger.info(f"  损失 (Actor): {avg_actor_loss:.4f}")
                    self.logger.info(f"  损失 (Critic, 总加权): {avg_critic_loss:.4f}")
                    self.logger.info(f"  --- 尺度监控 ---")
                    self.logger.info(f"  未加权 L_hv: {avg_L_hv_unweighted:.6f}") # (使用 .6f 提高精度)
                    self.logger.info(f"  -------------------")
                    # --- (!! 核心结束 !!) ---
                    
                    # 检查是否需要保存最佳模型
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        self._save_checkpoint(0, N, suffix="_best")
                        self.logger.info(f" *** 保存新的最佳模型! 平均奖励: {avg_reward:.4f} *** ")
                
                # --- (修改) 重置 Meters ---
                self.avg_reward.reset()
                self.avg_actor_loss.reset()
                self.avg_critic_loss_total.reset()
                self.avg_unweighted_hv_loss.reset()

            
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
        node_embeddings = state.node_embeddings
        
        # 2. Actor 生成动作
        with torch.no_grad():
            action, g_s = self.actor(h_graph, node_embeddings)
        
        # 4. 与环境交互
        next_state, reward, done, _ = self.env.step(action)
        
        # 5. 存储经验
        reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.tensor(done, dtype=torch.bool)
        
        # (buffer.push 会自动将 GPU 张量移到 CPU 存储)
        self.buffer.push(h_graph, g_s, node_embeddings, action, reward_tensor, done_tensor)
        
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
        s, g_s, node_embeddings, a, r_hv, d = self.buffer.sample(self.batch_size)
        
        # 将数据移到 GPU
        s = s.to(self.device)
        g_s = g_s.to(self.device)
        node_embeddings = node_embeddings.to(self.device)
        a = a.to(self.device)
        r_hv = r_hv.to(self.device)
        d = d.to(self.device)
        
        
        # 训练 Critic
        with torch.no_grad():
            y_hv = r_hv
            
        max_grad_norm = self.trainer_params.get('max_grad_norm_critic', 1.0)
        # 訓練 Critic 1
        q_current_1 = self.critic_1(node_embeddings, a, g_s)
        critic_loss_1_hv = F.mse_loss(q_current_1, y_hv)
        self.critic_1_optimizer.zero_grad()
        critic_loss_1_hv.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_grad_norm)
        self.critic_1_optimizer.step()
        
        # 訓練 Critic 2
        q_current_2 = self.critic_2(node_embeddings, a, g_s)
        critic_loss_2_hv = F.mse_loss(q_current_2, y_hv)
        self.critic_2_optimizer.zero_grad()
        critic_loss_2_hv.backward()
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_grad_norm)
        self.critic_2_optimizer.step()
        
        critic_loss_val = (critic_loss_1_hv.item() + critic_loss_2_hv.item()) / 2
        
        unweighted_hv_loss_val = (critic_loss_1_hv.item() + critic_loss_2_hv.item()) / 2
        
        actor_loss_val = 0.0
        
        # 训练 Actor
        if self.train_step_counter % 3 == 0:
            for p in self.critic_1.parameters(): p.requires_grad = False
            for p in self.critic_2.parameters(): p.requires_grad = False
                
            a_new, g_s_new = self.actor(s, node_embeddings)  # 只取action部分，忽略g_s
            q_score_1 = self.critic_1(node_embeddings, a_new, g_s_new)
            q_score_2 = self.critic_2(node_embeddings, a_new, g_s_new)
            q_hv_min = torch.min(q_score_1, q_score_2)
                        
            actor_loss = -q_hv_min.mean()
            actor_loss_val = actor_loss.item() # 保存
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 添加梯度裁剪
            self.actor_optimizer.step()
            
            for p in self.critic_1.parameters(): p.requires_grad = True
            for p in self.critic_2.parameters(): p.requires_grad = True
            
        # --- 输出 ---
        return actor_loss_val, critic_loss_val, unweighted_hv_loss_val
        

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
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }, current_checkpoint_path)
        
        self.logger.info(f"模型已保存到: {current_checkpoint_path}")
        return current_checkpoint_path