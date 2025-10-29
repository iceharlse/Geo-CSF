import torch
from logging import getLogger

# 导入环境和模型
from MOTSPEnv import TSPEnv as Env
from MOTSPModel import TSPModel as Model

# 导入优化器和学习率调度器
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

# 导入工具函数
from utils.utils import *

import numpy as np

class TSPTrainer:
    """
    TSP训练器类
    负责训练MOTSP模型，包括训练循环、损失计算、模型保存等功能
    """
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):
        """
        初始化训练器
        
        Args:
            env_params: 环境参数字典
            model_params: 模型参数字典
            optimizer_params: 优化器参数字典
            trainer_params: 训练器参数字典
        """

        # 保存参数
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # 结果文件夹和日志记录器
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # CUDA设置
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # 主要组件初始化
        self.model = Model(**self.model_params)  # 创建模型实例
        
        self.env = Env(**self.env_params)  # 创建环境实例
        # 创建优化器和学习率调度器
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # 模型恢复设置
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            # 如果启用模型加载，则从检查点恢复训练
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # 工具类实例化
        self.time_estimator = TimeEstimator()

    def run(self):
        """
        运行训练过程
        控制整个训练循环，包括学习率衰减、模型保存等
        """
        # 重置时间估计器
        self.time_estimator.reset(self.start_epoch)
        
        # 训练循环
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # 学习率衰减
            self.scheduler.step()

            # 训练一个epoch
            train_score_obj1, train_score_obj2, train_loss = self._train_one_epoch(epoch)
            # 记录训练结果
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # 日志记录和检查点保存
            ############################
            # 获取时间估计
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            # 判断是否完成训练
            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
       
            # 根据条件保存模型检查点
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_motsp-{}.pt'.format(self.result_folder, epoch))

            # 训练完成时的处理
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            tuple: (目标1得分, 目标2得分, 平均损失)
        """

        # 初始化评估指标
        score_AM_obj1 = AverageMeter()  # 目标1得分统计
        score_AM_obj2 = AverageMeter()  # 目标2得分统计
        loss_AM = AverageMeter()        # 损失统计

        # 获取训练episode数量
        train_num_episode = self.trainer_params['train_episodes']  #样本轮次，100,000
        episode = 0
        loop_cnt = 0
        
        # 训练循环
        while episode < train_num_episode:
            # 计算剩余episode数量和当前batch大小
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # 训练一个batch
            avg_score_obj1, avg_score_obj2, avg_loss = self._train_one_batch(batch_size)
            # 更新统计信息
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # 在第一个epoch时记录前10个batch的日志
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # 记录每个epoch的日志
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        """
        训练一个batch
        
        Args:
            batch_size: batch大小
            
        Returns:
            tuple: (目标1得分, 目标2得分, 平均损失)
        """

        # 准备阶段
        ###############################################
        self.model.train()  # 设置模型为训练模式
        self.env.load_problems(batch_size)  # 加载问题数据

        # 生成随机偏好向量（Dirichlet分布）
        alpha = 1
        pref = np.random.dirichlet((alpha, alpha), None)
        pref = torch.tensor(pref).float()

        # 重置环境状态
        reset_state, _, _ = self.env.reset()

        # 模型预前向传播（编码问题数据和偏好向量）
        self.model.pre_forward(reset_state, pref)
        
        # 初始化概率列表
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
      
        # POMO rollout过程
        ###############################################
        state, reward, done = self.env.pre_step()
        
        # 执行完整的路径构建过程
        while not done:
            # 模型选择下一个城市并获取选择概率
            selected, prob = self.model(state) #prob shape: (batch, pomo)
            # 执行选择动作并更新环境状态
            state, reward, done = self.env.step(selected)   
            # 记录选择概率
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2) 
            
        # 损失计算
        ###############################################
        # 奖励原本是负数，这里将其转换为正数以计算Tchebycheff奖励
        reward = - reward
        
        # 根据决策方法计算Tchebycheff奖励
        if self.trainer_params['dec_method'] == "WS":
            # 加权和方法
            tch_reward = (pref * reward).sum(dim=2)
        elif self.trainer_params['dec_method'] == "TCH":
            # Tchebycheff方法
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        else:
            return NotImplementedError
        
        # 将奖励重新设为负数
        reward = -reward
        tch_reward = -tch_reward

        # 计算对数概率
        log_prob = prob_list.log().sum(dim=2)
        # shape = (batch, group)
    
        # 计算优势函数（相对于平均奖励的偏差）
        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)
    
        # 计算策略梯度损失
        tch_loss = -tch_advantage * log_prob # 负号因为是最小化
        # shape = (batch, group)
        loss_mean = tch_loss.mean()
        
        # 得分计算
        ###############################################
        # 获取最优解的索引
        _ , max_idx = tch_reward.max(dim=1) #shape = (batch,)
        max_idx = max_idx.reshape(max_idx.shape[0],1) #shape = (batch,1)
        # 提取对应的目标得分
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)
        
        # 计算平均得分
        score_mean_obj1 = - max_reward_obj1.float().mean()
        score_mean_obj2 = - max_reward_obj2.float().mean()
    
        # 参数更新和返回
        ################################################
        # 清零梯度，反向传播，更新参数
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        # 返回结果
        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()