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
from MOTSP.POMO.MOTSProblemDef import get_random_problems, Reset_State
# 假设 cal_ps_hv 能处理批处理，如果不能，需要修改或替换
# from utils.cal_ps_hv import cal_ps_hv # 注意：原始cal_ps_hv可能不支持批处理HV计算
from dataclasses import dataclass

@dataclass
class State:
    """
    状态数据类，用于存储环境状态信息
    h_graph shape: (batch, embedding_dim)
    """
    h_graph: torch.Tensor

class MOCOEnv:
    """
    MOCO 强化学习环境 (支持批处理)。
    状态 s: 一批问题实例的 h_graph 嵌入。
    动作 a: 一批生成的偏好向量集 Λ。
    奖励 R: 每个实例对应的 Hypervolume。
    """

    def __init__(self, weca_model_params, weca_env_params, weca_checkpoint_path, batch_size=1, device='cpu', ref_point=None):
        """
        初始化MOCO环境

        Args:
            weca_model_params (dict): WE-CA 模型的参数。
            weca_env_params (dict): WE-CA 环境的参数 (主要是 problem_size, pomo_size)。
            weca_checkpoint_path (str): 冻结的 WE-CA 模型权重路径。
            batch_size (int): 环境期望处理的批次大小 (用于内部环境初始化)。
            device (str): 计算设备 ('cuda' or 'cpu')。
            ref_point (list or np.array): HV 计算的**固定**参考点 (例如 [20.0, 20.0])。
                                         **必须**提供以保证奖励信号稳定。
        """
        self.weca_model_params = weca_model_params
        self.weca_env_params = weca_env_params
        self.device = torch.device(device)
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
        self.current_problem_instances = None # 注意是复数
        self.current_h_graphs = None # 注意是复数

    def _calculate_batch_hv(self, batch_objectives, ref_point):
        """
        计算批处理的超体积指标 (Hypervolume)

        Args:
            batch_objectives (torch.Tensor): 批次的帕累托前沿解集
                shape: (batch, num_solutions, M)
            ref_point (np.array): 参考点
                shape: (M,)

        Returns:
            np.array: 每个实例的超体积值
                shape: (batch,)
        """
        batch_size = batch_objectives.size(0)
        num_solutions = batch_objectives.size(1)
        hv_values = np.zeros(batch_size)
        objectives_np = batch_objectives.detach().cpu().numpy()
        ref_point_np = ref_point # 已经是 numpy 了

        for i in range(batch_size):
            objectives_i = objectives_np[i] # shape: (num_solutions, M)

            # --- （之前的健壮性检查逻辑可以放在这里） ---
            if objectives_i.shape[0] == 0:
                hv_values[i] = 0.0
                continue

            if np.any(np.isnan(objectives_i)) or np.any(np.isinf(objectives_i)) or \
               np.any(np.isnan(ref_point_np)) or np.any(np.isinf(ref_point_np)):
                print(f"警告 (实例 {i}): objectives 或 ref_point 中存在 NaN 或 Inf。返回 HV=0。")
                hv_values[i] = 0.0
                continue

            if not np.all(np.all(ref_point_np >= objectives_i, axis=1)):
                 # print(f"警告 (实例 {i}): 参考点未能支配所有目标点。返回 HV=0。")
                 hv_values[i] = 0.0
                 continue
            # --- 健壮性检查结束 ---

            try:
                # hvwfg 需要 float64
                hv = hvwfg.wfg(objectives_i.astype(np.float64), ref_point_np.astype(np.float64))
                hv_values[i] = hv
            except Exception as e:
                print(f"计算 HV 时发生错误 (实例 {i}): {e}")
                hv_values[i] = 0.0 # 异常时返回0

        return hv_values # 返回 numpy 数组

    def reset(self, batch_problem_instances=None):
        """
        开始一个新回合。加载/生成一批问题实例，计算并返回初始状态 State(h_graphs)。

        Args:
            batch_problem_instances (torch.Tensor, optional): 提供一批特定的问题实例
                                                              (shape: [batch_size, problem_size, 4])。
                                                              如果为 None，则随机生成 self.batch_size 个。

        Returns:
            State: 包含一批初始状态 h_graphs 的 State 物件 (shape: [batch_size, embedding_dim])。
        """
        actual_batch_size = self.batch_size # 默认使用 PPO 的 batch_size

        if batch_problem_instances is None:
            # 生成 PPO batch_size 个随机问题实例
            batch_problem_instances = get_random_problems(actual_batch_size, self.problem_size).to(self.device)
        else:
            if not isinstance(batch_problem_instances, torch.Tensor):
                batch_problem_instances = torch.tensor(batch_problem_instances, dtype=torch.float32)
            if batch_problem_instances.shape != (actual_batch_size, self.problem_size, 4):
                 # 如果 PPO 传来的 batch_size 可能变化，这里需要处理
                 if batch_problem_instances.dim() == 3 and batch_problem_instances.shape[1] == self.problem_size and batch_problem_instances.shape[2] == 4:
                     actual_batch_size = batch_problem_instances.shape[0]
                     print(f"  接收到不同批次大小: {actual_batch_size}")
                 else:
                     raise ValueError(f"提供的 batch_problem_instances shape 错误 ({batch_problem_instances.shape})，应为 ({actual_batch_size}, {self.problem_size}, 4)")
            batch_problem_instances = batch_problem_instances.to(self.device)

        self.current_problem_instances = batch_problem_instances # 存储 (B, problem, 4)

        # --- 计算状态 h_graphs (批处理) ---
        with torch.no_grad():
            # 扩展 dummy_pref 以匹配批次大小
            dummy_pref_batch = torch.tensor([0.5, 0.5], device=self.device).repeat(actual_batch_size, 1) # (B, 2)
            encoded_nodes = self.encoder(self.current_problem_instances, dummy_pref_batch)
            # encoded_nodes shape: (B, problem_size + 1, embedding_dim)
            h_graphs_calculated = torch.mean(encoded_nodes[:, :-1, :], dim=1)
            # h_graphs_calculated shape: (B, embedding_dim)
            self.current_h_graphs = h_graphs_calculated # 存储计算出的 h_graphs

        return State(h_graph=self.current_h_graphs) # 返回包含 h_graphs 的 State 物件

    def step(self, batch_action_preference_sets):
        """
        执行一个批次的动作 (评估一批偏好集)，返回 (next_states, rewards, dones, infos)。

        Args:
            batch_action_preference_sets (torch.Tensor): Geo-CSF 生成的一批偏好向量集 Λ₁
                                                      (shape: [batch_size, N, M])。

        Returns:
            tuple: (next_states, rewards, dones, infos)
                   - next_states (State): 包含一批 h_graphs 的 State 物件 (与当前状态相同)。
                   - rewards (np.array): 每个实例计算得到的 Hypervolume (shape: [batch_size,])。
                   - dones (np.array): 每个实例是否完成 (总是 True, shape: [batch_size,])。
                   - infos (list[dict]): 每个实例的额外信息列表。
        """
        if self.current_problem_instances is None or self.current_h_graphs is None:
            raise RuntimeError("必须先调用 reset() 才能调用 step()")

        # 确保动作在正确的设备上
        batch_action_preference_sets = batch_action_preference_sets.to(self.device)

        B = self.current_h_graphs.shape[0] # 当前批次大小
        N = batch_action_preference_sets.shape[1] # 偏好向量数量
        M = batch_action_preference_sets.shape[2] # 目标数量

        # 检查输入的批次大小是否匹配
        if batch_action_preference_sets.shape[0] != B:
            raise ValueError(f"动作批次大小 ({batch_action_preference_sets.shape[0]}) 与状态批次大小 ({B}) 不匹配！")

        # 存储每个实例、每个偏好向量对应的最优解
        # 形状: (batch_size, N, M)
        all_batch_best_solutions = torch.zeros((B, N, M), device=self.device, dtype=torch.float32)

        pbar = tqdm(range(N), desc="  WE-CA 求解中 (所有偏好)", leave=False, disable=True)

        # 只需要加载一次问题实例到内部环境
        self.internal_env.load_problems(B, problems=self.current_problem_instances)

        # --- 循环处理每个偏好向量 λ_i，但对所有 B 个实例并行 ---
        for i in pbar:
            # current_prefs shape: (B, M) - 取出所有实例的第 i 个偏好
            current_prefs = batch_action_preference_sets[:, i, :]

            # --- 为当前偏好向量批次执行一次完整的 WE-CA 推理 ---
            # 内部环境状态需要重置
            reset_state_batch, _, _ = self.internal_env.reset() # reset 会处理批次 B

            # 模型预计算 (传入当前偏好批次)
            # pre_forward 需要 pref shape (B, M)
            self.solver_model.pre_forward(reset_state_batch, current_prefs)

            state_batch, reward_batch, done_batch = self.internal_env.pre_step()

            # 解码循环 (对批次 B 并行)
            with torch.no_grad():
                step_count = 0
                max_steps = self.problem_size + 1
                # done_batch 现在是 (B,) 或类似，需要检查 all()
                # 或者内部环境 step 返回的 done 是标量，表示所有都完成了？
                # 假设 internal_env.step 返回的 done 是标量 bool
                done_flag = False
                while not done_flag and step_count < max_steps:
                    # selected_batch shape: (B, weca_pomo_size)
                    selected_batch, _ = self.solver_model(state_batch)
                    state_batch, reward_batch, done_flag = self.internal_env.step(selected_batch)
                    step_count += 1
                if step_count >= max_steps:
                    print(f"警告: 偏好向量索引 {i} 的解码循环可能存在问题 (超時)。")
                    # 对于超时的批次项，需要特殊处理，例如标记为无效
                    # reward_batch 可能为 None 或部分有效

            # --- 解码完成，选择每个实例的最佳解 ---
            if reward_batch is not None:
                # final_costs_batch shape: (B, weca_pomo_size, M)
                final_costs_batch = -reward_batch

                # 使用当前偏好批次 current_prefs (shape: B, M) 计算标量成本
                # 需要扩展 pref: (B, 1, M)
                scalarized_costs_batch = (current_prefs.unsqueeze(1) * final_costs_batch).sum(dim=-1)
                # scalarized_costs_batch shape: (B, weca_pomo_size)

                # 找到每个实例中成本最低的解的索引
                # best_pomo_indices shape: (B,)
                best_pomo_indices = torch.argmin(scalarized_costs_batch, dim=1)

                # 提取每个实例的最佳解的原始多目标成本
                # 使用 advanced indexing 或 gather
                # gather 需要 index shape (B, 1, M)
                best_pomo_indices_expanded = best_pomo_indices.view(B, 1, 1).expand(-1, -1, M)
                # best_solutions_batch shape: (B, 1, M) -> (B, M) after squeeze
                best_solutions_batch = torch.gather(final_costs_batch, 1, best_pomo_indices_expanded).squeeze(1)

                # 存储这批结果
                all_batch_best_solutions[:, i, :] = best_solutions_batch
            else:
                print(f"警告: 偏好向量索引 {i} 未能为所有实例生成有效解。")
                # 可以用一个特殊值（如 Inf 或 NaN）填充 all_batch_best_solutions[:, i, :]
                all_batch_best_solutions[:, i, :] = float('inf')

            # --- 当前偏好处理结束 ---

        # --- 计算 HV 奖励 (批处理) ---
        # all_batch_best_solutions shape: (B, N, M)
        # 需要处理可能存在的 Inf 值 (无效解)
        # 例如，对每个实例，只保留有效的解来计算 HV
        hv_rewards_np = self._calculate_batch_hv(all_batch_best_solutions, self.ref_point)
        # hv_rewards_np shape: (B,)

        # done 永远是 True (对于每个实例)
        dones_np = np.ones(B, dtype=bool)

        # next_states 就是当前的 h_graphs
        next_states = State(h_graph=self.current_h_graphs) # shape: (B, embedding_dim)

        # infos 列表，每个元素是一个字典
        infos = [{'objectives': all_batch_best_solutions[b].cpu().numpy()} for b in range(B)]

        return next_states, hv_rewards_np, dones_np, infos

# --- (测试代码需要相应修改以适应批处理) ---
if __name__ == '__main__':
    # 示例用法
    B = 4 # 测试批次大小
    model_params = { 'embedding_dim': 128, 'encoder_layer_num': 6, 'head_num': 8, 'qkv_dim': 16, 'ff_hidden_dim': 512, 'eval_type': 'argmax'}
    env_params = { 'problem_size': 20, 'pomo_size': 20 } # 这里的pomo_size是WE-CA内部的
    checkpoint_path = 'result/train__tsp_n20/checkpoint_motsp-200.pt' # 您的模型路径
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ref_point = [20.0, 20.0]

    # 初始化环境时传入 PPO 的批次大小
    env = MOCOEnv(model_params, env_params, checkpoint_path, batch_size=B, device=device, ref_point=ref_point)

    # --- 测试 reset ---
    initial_state_batch = env.reset() # 随机生成 B 个问题
    print("Reset 完成，初始状态 h_graphs shape:", initial_state_batch.h_graph.shape) # 应为 (B, embed_dim)

    # --- 测试 step ---
    N = 10 # 生成10个偏好
    # 为批次中的每个实例生成一个偏好集
    prefs_batch_raw = torch.rand(B, N, 2)
    prefs_batch = prefs_batch_raw / prefs_batch_raw.sum(dim=2, keepdim=True) # shape: (B, N, 2)

    next_states, rewards, dones, infos = env.step(prefs_batch)

    print(f"Step 完成:")
    print(f"  Next States h_graphs shape: {next_states.h_graph.shape}") # 应为 (B, embed_dim)
    print(f"  HV rewards shape: {rewards.shape}") # 应为 (B,)
    print(f"  Dones shape: {dones.shape}") # 应为 (B,)
    print(f"  Infos length: {len(infos)}") # 应为 B
    print(f"  Info[0] objectives shape: {infos[0]['objectives'].shape}") # 应为 (N, 2) or (<=N, 2)
    print(f"  Example HV rewards: {rewards[:min(B, 5)]}") # 打印前几个奖励值