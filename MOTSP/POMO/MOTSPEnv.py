##############################################
# 多目标旅行商问题(MOTSP)环境类
# 实现了POMO(Population-based Monte Carlo)算法的环境
##############################################

# 导入必要的库
from dataclasses import dataclass  # 用于创建数据类
import torch  # PyTorch深度学习框架

# 导入问题定义模块中的函数
# get_random_problems: 生成随机TSP问题
# augment_xy_data_by_64_fold_2obj: 数据增强函数，将数据扩充64倍
from MOTSP.POMO.MOTSProblemDef import get_random_problems, augment_xy_data_by_64_fold_2obj


##############################################
# 数据类定义
##############################################

@dataclass
class Reset_State:
    """
    环境重置状态数据类
    用于存储环境重置后的初始状态信息
    """
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    # problems: 批次中每个问题的城市坐标信息
    # 维度说明:
    # - batch: 批次大小
    # - problem: 问题规模(城市数量)
    # - 2: 城市的x,y坐标


@dataclass
class Step_State:
    """
    环境步骤状态数据类
    用于存储环境每一步的状态信息
    """
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    # BATCH_IDX: 批次索引，用于标识不同的问题实例
    # POMO_IDX: POMO索引，用于标识同一个问题的不同解
    
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    # current_node: 当前访问的城市节点
    
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)
    # ninf_mask: 无效动作掩码，-inf表示不可访问的城市


##############################################
# TSP环境类定义
##############################################

class TSPEnv:
    """
    多目标TSP环境类
    实现了POMO算法的环境接口，用于训练和测试多目标TSP求解器
    """
    
    def __init__(self, **env_params):
        """
        初始化TSP环境
        
        Args:
            **env_params: 环境参数字典
                - problem_size: 问题规模(城市数量)
                - pomo_size: POMO种群大小(每个问题生成的不同解的数量)
        """
        
        # 常量初始化 @INIT
        ####################################
        self.env_params = env_params  # 存储环境参数
        self.problem_size = env_params['problem_size']  # 问题规模(城市数量)
        self.pomo_size = env_params['pomo_size']  # POMO种群大小

        # 常量初始化 @Load_Problem
        ####################################
        self.batch_size = None  # 批次大小
        self.BATCH_IDX = None  # 批次索引张量
        self.POMO_IDX = None  # POMO索引张量
        # IDX.shape: (batch, pomo)
        self.problems = None  # 问题数据张量
        # shape: (batch, node, node)

        # 动态变量初始化
        ####################################
        self.selected_count = None  # 已选择的城市数量
        self.current_node = None  # 当前节点
        # shape: (batch, pomo)
        self.selected_node_list = None  # 已选择的节点列表
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1, problems=None):
        """
        加载TSP问题数据
        
        Args:
            batch_size: 批次大小
            aug_factor: 数据增强因子，默认为1(不增强)
            problems: 预定义的问题数据，如果为None则随机生成
            
        Returns:
            None
        """
        self.batch_size = batch_size  # 设置批次大小
        
        # 如果提供了问题数据则使用，否则随机生成
        if problems is not None:
            self.problems = problems
        else:
            # 调用问题定义模块生成随机问题
            self.problems = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)
        
        # 数据增强处理
        if aug_factor > 1:
            if aug_factor == 64:
                # 64倍数据增强
                self.batch_size = self.batch_size * 64
                # 调用数据增强函数
                self.problems = augment_xy_data_by_64_fold_2obj(self.problems)
            else:
                # 其他增强因子未实现
                raise NotImplementedError

        # 创建批次索引和POMO索引矩阵
        # BATCH_IDX: 每行都是[0,1,2,...,batch_size-1]重复pomo_size次
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        # POMO_IDX: 每列都是[0,1,2,...,pomo_size-1]重复batch_size次
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        """
        重置环境到初始状态
        
        Returns:
            tuple: (reset_state, reward, done)
                - reset_state: 重置状态对象
                - reward: 奖励值(初始为None)
                - done: 是否完成标志(初始为False)
        """
        self.selected_count = 0  # 已选择城市数清零
        self.current_node = None  # 当前节点清空
        # shape: (batch, pomo)
        
        # 初始化已选择节点列表为空列表
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # 创建步骤状态对象
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        # 初始化无效动作掩码为全0(所有城市都可访问)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None  # 初始奖励为None
        done = False   # 初始未完成
        # 返回重置状态、奖励和完成标志
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        """
        步骤前准备函数
        
        Returns:
            tuple: (step_state, reward, done)
                - step_state: 当前步骤状态
                - reward: 奖励值
                - done: 是否完成标志
        """
        reward = None  # 奖励值为None
        done = False   # 未完成
        # 返回当前步骤状态、奖励和完成标志
        return self.step_state, reward, done

    def step(self, selected):
        """
        执行一步动作(选择一个城市)
        
        Args:
            selected: 选择的城市索引
                shape: (batch, pomo)
                
        Returns:
            tuple: (step_state, reward, done)
                - step_state: 更新后的步骤状态
                - reward: 奖励值(完成时计算旅行距离)
                - done: 是否完成标志
        """
        # selected.shape: (batch, pomo)

        self.selected_count += 1  # 增加已选择城市计数
        self.current_node = selected  # 更新当前节点
        # shape: (batch, pomo)
        
        # 将当前选择的节点添加到已选择节点列表
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # 更新步骤状态
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        
        # 更新无效动作掩码，将已选择的城市标记为不可访问(-inf)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # 检查是否完成(所有城市都已访问)
        done = (self.selected_count == self.problem_size)
        
        # 如果完成，计算奖励(负的旅行距离)
        if done:
            reward = -self._get_travel_distance()  # 注意负号!
        else:
            reward = None  # 未完成时奖励为None

        # 返回更新后的步骤状态、奖励和完成标志
        return self.step_state, reward, done

    def _get_travel_distance(self):
        """
        计算旅行距离(多目标版本)
        计算每个POMO解的两个目标的旅行距离
        
        Returns:
            torch.Tensor: 旅行距离向量
                shape: (batch, pomo, 2)
        """
        # 创建gather索引，用于重新排列城市访问顺序
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 4)
        # shape: (batch, pomo, problem, 4)
        
        # 扩展问题数据以匹配POMO维度
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 4)

        # 根据访问顺序重新排列城市坐标
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 4)
        
        # 循环移位，用于计算相邻城市间的距离
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        # 计算第一个目标(通常是物理距离)的线段长度
        # 使用前两个坐标(x1,y1)计算欧几里得距离
        segment_lengths_obj1 = ((ordered_seq[:, :, :, :2]-rolled_seq[:, :, :, :2])**2).sum(3).sqrt()
        
        # 计算第二个目标(通常是成本)的线段长度
        # 使用后两个坐标(x2,y2)计算欧几里得距离
        segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:]-rolled_seq[:, :, :, 2:])**2).sum(3).sqrt()

        # 计算总旅行距离(所有线段长度之和)
        travel_distances_obj1 = segment_lengths_obj1.sum(2)  # 第一个目标总距离
        travel_distances_obj2 = segment_lengths_obj2.sum(2)  # 第二个目标总距离
    
        # 将两个目标的距离组合成向量
        travel_distances_vec = torch.stack([travel_distances_obj1,travel_distances_obj2],axis = 2)
        
        # shape: (batch, pomo, 2)
        return travel_distances_vec