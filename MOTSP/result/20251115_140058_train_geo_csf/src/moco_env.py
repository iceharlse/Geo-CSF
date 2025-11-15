import torch
import torch.nn as nn
import numpy as np
import hvwfg
import os
import sys
from tqdm import tqdm # 可选导入

# --- 确保能导入模块 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MOTSP.POMO.MOTSPEnv import TSPEnv as MOTSPEnv
from MOTSP.POMO.MOTSPModel import TSPModel as MOTSPModel
from MOTSP.MOTSProblemDef import get_random_problems
from utils.cal_ps_hv import cal_ps_hv 
from utils.cal_pareto_demo import Pareto_sols
from dataclasses import dataclass

@dataclass
class State:
    """
    状态数据类，用于存储环境状态信息
    h_graph shape: (batch, embedding_dim)
    """
    h_graph: torch.Tensor = None
    node_embeddings: torch.Tensor = None

class MOCOEnv:
    """
    MOCO 强化学习环境 (支持批处理)。
    状态 s: 一批问题实例的 h_graph 嵌入。
    动作 a: 一批生成的偏好向量集 Λ。
    奖励 R: 每个实例对应的 Hypervolume。
    """

    def __init__(self, **moco_env_params):
        """
        初始化MOCO环境

        Args:
            moco_env_params (dict): MOCO环境的参数字典，包含以下键：
                - weca_model_params (dict): WE-CA 模型的参数。
                - weca_env_params (dict): WE-CA 环境的参数 (主要是 problem_size, pomo_size)。
                - weca_checkpoint_path (str): 冻结的 WE-CA 模型权重路径。
                - batch_size (int): 环境期望处理的批次大小 (用于内部环境初始化)。
                - device (str): 计算设备 ('cuda' or 'cpu')。
                - ref_point (list or np.array): HV 计算的**固定**参考点 (例如 [20.0, 20.0])。
        """
        # 从参数字典中提取各个参数
        weca_model_params = moco_env_params['weca_model_params']
        weca_env_params = moco_env_params['weca_env_params']
        weca_checkpoint_path = moco_env_params['weca_checkpoint_path']
        batch_size = moco_env_params.get('batch_size', 1)
        device = moco_env_params.get('device', 'cuda')
        ref_point = moco_env_params.get('ref_point', None)
        
        self.weca_model_params = weca_model_params
        self.weca_env_params = weca_env_params
        self.device = torch.device(device)
        
        # 它会影响所有后续的张量创建
        # 尤其是 self.internal_env 内部创建的张量
        if self.device.type == 'cuda':
            # 设置默认设备为 GPU
            torch.set_default_device(self.device) 
            # 设置默认浮点类型（通常是 float32）
            torch.set_default_dtype(torch.float32) 
            print(f"MOCOEnv: 已设置默认设备为 {self.device}，默认类型为 torch.float32")
        else:
            # 设置默认设备为 CPU
            torch.set_default_device('cpu') 
            # 设置默认浮点类型
            torch.set_default_dtype(torch.float32) 
            print("MOCOEnv: 已设置默认设备为 cpu，默认类型为 torch.float32")
        # --- 新增结束 ---
        
                
        self.problem_size = weca_env_params['problem_size']
        self.weca_pomo_size = weca_env_params['pomo_size'] # WE-CA 内部的 POMO size
        self.batch_size = batch_size # PPO希望处理的批次大小

        # --- 加载并冻结 WE-CA 模型 ---
        print("加载冻结的 WE-CA 模型...")
        self.solver_model = MOTSPModel(**weca_model_params).to(self.device)
        self.solver_model.eval()
        try:
            checkpoint = torch.load(weca_checkpoint_path, map_location=self.device)
            self.solver_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载 WE-CA 检查点: {weca_checkpoint_path}")
        except FileNotFoundError:
            print(f"错误: 找不到 WE-CA 检查点文件: {weca_checkpoint_path}")
            raise
        except Exception as e:
            print(f"加载 WE-CA 检查点时出错: {e}")
            raise

        for param in self.solver_model.parameters():
            param.requires_grad = False
        print("WE-CA 模型参数已冻结。")

        # 提取冻结的 Encoder (用于计算 h_graph)
        self.encoder = self.solver_model.encoder
        self.encoder.eval()

        # --- 初始化内部环境 (使用PPO的batch_size) ---
        # 注意：这里的 batch_size 是 PPO 的 batch_size
        self.internal_env = MOTSPEnv(**{**weca_env_params, 'batch_size': self.batch_size})

        # --- HV 计算相关 ---
        if ref_point is None:
            raise ValueError("必须为 MOCOEnv 提供一个固定的 ref_point 用于 HV 计算！")
        self.ref_point = np.array(ref_point, dtype=float)
        print(f"使用固定的 HV 参考点: {self.ref_point}")

        # --- 存储当前状态 ---
        self.problems = None  
        self.h_graphs = None  
        self.node_embeddings = None

    def reset(self, problem=None):
        """
        开始一个新回合。加载/生成一批问题实例，计算并返回初始状态 State(h_graphs)。

        Args:
            problem_instances (torch.Tensor, optional): 提供一批特定的问题实例
                                                              (shape: [batch_size, problem_size, 4])。
                                                              如果为 None，则随机生成 self.batch_size 个。

        Returns:
            State: 包含一批初始状态 h_graphs 的 State 物件 (shape: [batch_size, embedding_dim])。
        """
        actual_batch_size = self.batch_size # 默认使用 PPO 的 batch_size

        if problem is None:
            # 生成 PPO batch_size 个随机问题实例
            problem = get_random_problems(actual_batch_size, self.problem_size).to(self.device)
        else:
            if not isinstance(problem, torch.Tensor):
                problem = torch.tensor(problem, dtype=torch.float32)
            if problem.shape != (actual_batch_size, self.problem_size, 4):
                 # 如果 PPO 传来的 batch_size 可能变化，这里需要处理
                 if problem.dim() == 3 and problem.shape[1] == self.problem_size and problem.shape[2] == 4:
                     actual_batch_size = problem.shape[0]
                     print(f"  接收到不同批次大小: {actual_batch_size}")
                 else:
                     raise ValueError(f"提供的 problem_instances shape 错误 ({problem.shape})，应为 ({actual_batch_size}, {self.problem_size}, 4)")
            problem = problem.to(self.device)

        self.problems = problem # 简化变量名 (B, problem, 4)

        # --- 计算状态 h_graphs (批处理) ---
        with torch.no_grad():
            # 扩展 dummy_pref 以匹配批次大小
            dummy_pref = torch.tensor([0.5, 0.5], device=self.device).repeat(actual_batch_size, 1) # (B, 2)
            encoded_nodes = self.encoder(self.problems, dummy_pref)
            # encoded_nodes shape: (B, problem_size + 1, embedding_dim)
            h_graphs = torch.mean(encoded_nodes[:, :-1, :], dim=1)
            # h_graphs shape: (B, embedding_dim)
            self.h_graphs = h_graphs
            self.node_embeddings = encoded_nodes[:, :-1, :] # (B, problem_size, embedding_dim)

        return State(h_graph=self.h_graphs, node_embeddings=self.node_embeddings)

    def _solve_with_preference(self, prefs):
        """
        使用给定的偏好向量求解问题
        
        Args:
            prefs (torch.Tensor): 偏好向量 (shape: [B, M])
            
        Returns:
            torch.Tensor: 最佳解的成本 (shape: [B, M])
        """
        B = prefs.shape[0]  # 批次大小
        M = prefs.shape[1]  # 目标数量
        
        # --- 为当前偏好向量批次执行一次完整的 WE-CA 推理 ---
        # 内部环境状态需要重置
        reset_state, _, _ = self.internal_env.reset()

        # 将问题实例和当前偏好传递给 WE-CA 模型进行encoding和预处理
        self.solver_model.pre_forward(reset_state, prefs)

        state, reward, done = self.internal_env.pre_step()

        # 解码循环 (对批次 B 并行)
        with torch.no_grad():
            step_count = 0
            max_steps = self.problem_size + 1  # 安全上限，防止无限循环

            done_flag = False
            while not done_flag and step_count < max_steps:
                # selected shape: (B, weca_pomo_size)
                selected, _ = self.solver_model(state)
                state, reward, done_flag = self.internal_env.step(selected) #reward是负的成本
                step_count += 1

        # --- 解码完成，选择每个实例的最佳解 ---
        if reward is not None:
            # costs shape: (B, weca_pomo_size, M)
            costs = -reward # 转为正的成本

            # 使用当前偏好批次 prefs (shape: B, M) 计算标量成本
            # 需要扩展 pref: (B, 1, M)
            scalarized = (prefs.unsqueeze(1) * costs).sum(dim=-1)
            # scalarized shape: (B, weca_pomo_size)

            # 找到每个实例中成本最低的解的索引
            # best_pomo_indices shape: (B,)
            best_pomo_indices = torch.argmin(scalarized, dim=1)

            # 提取每个实例的最佳解的原始多目标成本
            # 使用 advanced indexing 或 gather
            # gather 需要 index shape (B, 1, M)
            best_pomo_indices_expanded = best_pomo_indices.view(B, 1, 1).expand(-1, -1, M)
            # best_solutions shape: (B, 1, M) -> (B, M) after squeeze
            best_solutions = torch.gather(costs, 1, best_pomo_indices_expanded).squeeze(1)

            return best_solutions
        else:
            # 返回一个特殊值（如 Inf）表示无效解
            return torch.full((B, M), float('inf'), device=self.device)

    def _compute_pareto_solutions(self, solutions, N, M):
        """
        计算帕累托解集
        
        Args:
            solutions (torch.Tensor): 所有解 (shape: [B, N, M])
            N (int): 偏好向量数量
            M (int): 目标数量
            
        Returns:
            tuple: (p_sols, p_sols_num)
        """
        # 临时将默认类型重置为 CPU
        # 这样 Pareto_sols 内部创建的张量（如 BATCH_IDX）才会在 CPU 上
        saved_device = torch.get_default_device() # 保存当前设备 (GPU)
        torch.set_default_device('cpu') # 临时切换到 CPU
        
        B = solutions.shape[0]  # 批次大小
        
        # --- 计算帕累托解 ---
        # 创建 Pareto_sols 实例来计算每个批次的帕累托解
        nd_sort = Pareto_sols(p_size=self.problem_size, pop_size=N, obj_num=M)
        
        # 更新帕累托解集
        nd_sort.update_PE(objs=solutions)
        
        # 获取帕累托解集和数量
        p_sols, p_sols_num, _ = nd_sort.show_PE()
        
        # 恢复全局默认类型（GPU）
        torch.set_default_device(saved_device) # 恢复 GPU
        
        return p_sols, p_sols_num

    def _compute_hv_rewards(self, p_sols, p_sols_num):
        """
        计算 HV 奖励
        
        Args:
            p_sols: 帕累托解集
            p_sols_num: 帕累托解数量
            
        Returns:
            np.array: HV 奖励 (shape: [B,])
        """
        # 临时将默认类型重置为 CPU
        saved_device = torch.get_default_device() # 保存当前设备 (GPU)
        torch.set_default_device('cpu') # 临时切换到 CPU
        
        # --- 计算 HV 奖励 ---
        # 使用帕累托解来计算 HV
        hv_rewards_batch = cal_ps_hv(pf=p_sols, 
                                     pf_num=p_sols_num, 
                                     ref=self.ref_point, 
                                     ideal=np.array([0.0, 0.0]))
    
        # cal_ps_hv 返回的 shape 是 (B, 1)，我们需要 (B,)
        hv_rewards = hv_rewards_batch.squeeze(-1)
        
        # 恢复全局默认类型（GPU）
        torch.set_default_device(saved_device) # 恢复 GPU
        
        return hv_rewards

    def step(self, preference_sets):
        """
        执行一个批次的动作 (评估一批偏好集)，返回 (next_states, rewards, dones, infos)。
    
        Args:
            preference_sets (torch.Tensor): PPO Agent 生成的一批偏好向量集 Λ
                                                      (shape: [batch_size, N, M])。
    
        Returns:
            tuple: (next_states, rewards, dones, infos)
                   - next_states (State): 包含一批 h_graphs 的 State 物件 (与当前状态相同)。
                   - rewards (np.array): 每个实例计算得到的 Hypervolume (shape: [batch_size,])。
                   - dones (np.array): 每个实例是否完成 (总是 True, shape: [batch_size,])。
                   - infos (list[dict]): 每个实例的额外信息列表。
        """
        if self.problems is None or self.h_graphs is None:
            raise RuntimeError("必须先调用 reset() 才能调用 step()")
    
        # 确保动作在正确的设备上
        preference_sets = preference_sets.to(self.device)
    
        B = self.h_graphs.shape[0]  # 当前批次大小
        N = preference_sets.shape[1]  # 偏好向量数量
        M = preference_sets.shape[2]  # 目标数量
    
        # 检查输入的批次大小是否匹配
        if preference_sets.shape[0] != B:
            raise ValueError(f"动作批次大小 ({preference_sets.shape[0]}) 与状态批次大小 ({B}) 不匹配！")
    
        # 存储每个实例、每个偏好向量对应的最优解
        # 形状: (batch_size, N, M)
        solutions = torch.zeros((B, N, M), device=self.device, dtype=torch.float32)
    
        # 只需要加载一次问题实例到内部环境
        self.internal_env.load_problems(B, problems=self.problems)
    
        # --- 循环处理每个偏好向量 λ_i，但对所有 B 个实例并行 ---
        for i in range(N):
            # prefs shape: (B, M) - 取出所有实例的第 i 个偏好
            prefs = preference_sets[:, i, :]
            
            # 使用偏好向量求解问题
            best_solutions = self._solve_with_preference(prefs)
            
            # 存储这批结果
            solutions[:, i, :] = best_solutions

        # 在计算 HV 之前，将解移回 CPU
        solutions_cpu = solutions.cpu()
        
        # 计算帕累托解集
        p_sols, p_sols_num = self._compute_pareto_solutions(solutions_cpu, N, M)

        # 计算 HV 奖励
        hv_rewards = self._compute_hv_rewards(p_sols, p_sols_num)
        
        # 获取 NDS 奖励 (NDS 就是 p_sols_num)
        #    (p_sols_num 是 GPU 張量，我們需要將其轉為 CPU NumPy)
        nds_rewards = p_sols_num.cpu().numpy() # shape: (B,)
    
        # done 永远是 True (对于每个实例)
        dones = np.ones(B, dtype=bool)
    
        # next_states 就是当前的 h_graphs 和 node_embeddings
        next_states = State(h_graph=self.h_graphs, node_embeddings=self.node_embeddings)  # shape: (B, embedding_dim)
    
        # infos 列表，每个元素是一个字典
        infos = [{'objectives': solutions_cpu[b].numpy()} for b in range(B)]
    
        return next_states, hv_rewards, dones, infos

# --- (测试代码需要相应修改以适应批处理) ---
if __name__ == '__main__':
    # 示例用法
    B = 4 # 测试批次大小
    model_params = { 
        'embedding_dim': 128, 
        'encoder_layer_num': 6, 
        'head_num': 8, 
        'qkv_dim': 16, 
        'ff_hidden_dim': 512, 
        'eval_type': 'argmax',
        'sqrt_embedding_dim': 16,  # 添加缺失的参数
        'logit_clipping': 10.0     # 添加缺失的参数
    }
    env_params = { 'problem_size': 20, 'pomo_size': 20 } # 这里的pomo_size是WE-CA内部的
    
    # 修复模型路径：使用正确的相对路径指向POMO目录下的检查点文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, 'POMO', 'result', 'train__tsp_n20', 'checkpoint_motsp-200.pt')
    
    # 添加文件存在检查
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件不存在: {checkpoint_path}")
        print("请确保您已经训练了WE-CA模型或下载了预训练模型")
        # 如果找不到文件，可以尝试其他可能的路径
        alternative_path = os.path.join(script_dir, 'POMO', 'result', 'train__tsp_n20', 'checkpoint_motsp-200.pt')
        if os.path.exists(alternative_path):
            checkpoint_path = alternative_path
            print(f"使用备选路径: {checkpoint_path}")
        else:
            print("无法找到有效的检查点文件")
            sys.exit(1)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ref_point = [20.0, 20.0]

    # 构建moco_env_params字典
    moco_env_params = {
        'weca_model_params': model_params,
        'weca_env_params': env_params,
        'weca_checkpoint_path': checkpoint_path,
        'batch_size': B,
        'device': device,
        'ref_point': ref_point
    }

    # 初始化环境时传入 PPO 的批次大小
    env = MOCOEnv(**moco_env_params)

    # --- 测试 reset ---
    initial_state = env.reset() # 随机生成 B 个问题
    print("Reset 完成，初始状态 h_graphs shape:", initial_state.h_graph.shape) # 应为 (B, embed_dim)

    # --- 测试 step ---
    N = 10 # 生成10个偏好
    # 为批次中的每个实例生成一个偏好集
    prefs_raw = torch.rand(B, N, 2)
    prefs = prefs_raw / prefs_raw.sum(dim=2, keepdim=True) # shape: (B, N, 2)

    next_states, rewards, dones, infos = env.step(prefs)

    print(f"Step 完成:")
    print(f"  Next States h_graphs shape: {next_states.h_graph.shape}") # 应为 (B, embed_dim)
    print(f"  HV rewards shape: {rewards.shape}") # 应为 (B,)
    print(f"  Dones shape: {dones.shape}") # 应为 (B,)
    print(f"  Infos length: {len(infos)}") # 应为 B
    print(f"  Info[0] objectives shape: {infos[0]['objectives'].shape}") # 应为 (N, 2) or (<=N, 2)
    print(f"  Example HV rewards: {rewards[:min(B, 5)]}") # 打印前几个奖励值