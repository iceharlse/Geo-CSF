import torch

import os
from logging import getLogger

# 导入自定义模块
from MOTSPEnv import TSPEnv as Env  # 环境类
from MOTSPModel import TSPModel as Model  # 模型类

# 导入问题定义模块
from MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj

# 导入einops库用于张量操作
from einops import rearrange

# 导入工具函数
from utils.utils import *

# TSP测试器类
class TSPTester:
    def __init__(self,
                 env_params,  # 环境参数字典
                 model_params,  # 模型参数字典
                 tester_params,  # 测试器参数字典
                 logger=None,  # 日志记录器
                 result_folder=None,  # 结果保存文件夹
                 checkpoint_dict=None,  # 检查点字典
                 ):
        """
        初始化TSP测试器
        """

        # 保存传入的参数
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # 设置日志记录器和结果文件夹
        if logger:
            self.logger = logger
            self.result_folder = result_folder
        else:
            self.logger = getLogger(name='tester')
            self.result_folder = get_result_folder()

        # 配置CUDA设备
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # 创建环境和模型实例
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # 加载模型检查点
        if checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        else:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # 创建时间估算器
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref, episode=0):
        """
        运行测试过程
        :param shared_problem: 共享的问题数据
        :param pref: 偏好向量
        :param episode: 起始episode编号
        :return: 测试得分列表
        """
        self.time_estimator.reset()
    
        # 初始化平均得分记录器（针对两个目标）
        aug_score_AM = {}
        for i in range(2):  # 2个目标函数
            aug_score_AM[i] = AverageMeter()
            
        # 获取测试episode数量
        test_num_episode = self.tester_params['test_episodes']
        episode = episode
        
        # 循环测试所有episode
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # 测试一个批次
            aug_score = self._test_one_batch(shared_problem, pref, batch_size, episode)
            
            # 更新两个目标的平均得分
            for i in range(2):
                aug_score_AM[i].update(aug_score[i], batch_size)

            episode += batch_size
           
            # 记录日志信息
            self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(
                aug_score_AM[0].avg.mean(), aug_score_AM[1].avg.mean()))
            break # 仅测试一个批次
            
        # 返回测试结果
        return [aug_score_AM[0].avg.cpu(), aug_score_AM[1].avg.cpu()]
                
    def _test_one_batch(self, shared_probelm, pref, batch_size, episode):
        """
        测试一个批次的数据
        :param shared_probelm: 共享的问题数据（注意这里拼写有误，应该是shared_problem）
        :param pref: 偏好向量
        :param batch_size: 批次大小
        :param episode: episode编号
        :return: 测试得分
        """

        # 数据增强设置
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # 设置环境的批次大小和问题数据
        self.env.batch_size = batch_size
        self.env.problems = shared_probelm[episode: episode + batch_size]

        # 如果启用数据增强且增强因子为64，则进行数据增强
        if aug_factor == 64:
            self.env.batch_size = self.env.batch_size * 64
            self.env.problems = augment_xy_data_by_64_fold_2obj(self.env.problems)

        # 创建批次和POMO索引矩阵
        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)
        
        # 设置模型为评估模式
        self.model.eval()
        with torch.no_grad():
            # 重置环境
            reset_state, _, _ = self.env.reset()

            # 模型预前向传播
            self.model.pre_forward(reset_state, pref)
            
        # 环境预步骤
        state, reward, done = self.env.pre_step()
        
        # 执行完整的TSP路径构建过程
        while not done:
            # 模型选择下一个节点
            selected, _ = self.model(state)
            # 执行步骤
            state, reward, done = self.env.step(selected)
        
        # 奖励处理（将负奖励转为正奖励以便计算TCH）
        reward = - reward
        
        # 根据决策方法计算TCH奖励
        if self.tester_params['dec_method'] == "WS":
            # 加权和方法
            tch_reward = (pref * reward).sum(dim=2)
        elif self.tester_params['dec_method'] == "TCH":
            # Tchebycheff方法
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        else:
            return NotImplementedError

        # 将奖励恢复为负值
        reward = -reward
        tch_reward = -tch_reward
    
        # 重塑奖励张量
        tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        
        # 重新排列张量以整合增强和POMO维度
        tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)')  #相当于一个问题对应aug_factor*pomo_size个解
        _ , max_idx_aug = tch_reward_aug.max(dim=1)
        max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0],1) #shape: (batch_size,1)
        
        # 提取最优解的目标值
        max_reward_obj1 = rearrange(
            reward[:,:,0].reshape(aug_factor, batch_size, self.env.pomo_size), 
            'c b h -> b (c h)'
        ).gather(1, max_idx_aug)
        
        max_reward_obj2 = rearrange(
            reward[:,:,1].reshape(aug_factor, batch_size, self.env.pomo_size), 
            'c b h -> b (c h)'
        ).gather(1, max_idx_aug)
     
        # 构建返回的得分列表
        aug_score = []
        aug_score.append(-max_reward_obj1.float())
        aug_score.append(-max_reward_obj2.float())
        
        return aug_score