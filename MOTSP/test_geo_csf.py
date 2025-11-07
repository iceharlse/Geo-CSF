##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 5

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "./POMO")  # 添加POMO目录到路径中

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

# 导入自定义模块
from actor import MocoPolicyNetwork
from moco_env import MOCOEnv
from POMO.MOTSPTester import TSPTester as Tester

##########################################################################################
import time

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
    'weca_model_params': {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'hyper_hidden_dim': 256,
    },
    'weca_env_params': {
        'problem_size': 20,
        'pomo_size': 20
    },
    'weca_checkpoint_path': './POMO/result/train__tsp_n20/checkpoint_motsp-200.pt',
    'ref_point': [20.0, 20.0]
}

# Actor模型参数
actor_params = {
    'gfp_params': {
        'input_dim': 128,
        'hidden_dims': [64, 32],
        'output_dim': 16,
        'activation': 'relu'
    },
    'csf_params': {
        'input_dim': 2, # M
        'hidden_dim': 128,
        'condition_dim': 128, 
        'geometric_dim': 16,
        'time_embed_dim': 128,
        'num_layers': 2,
        'num_heads': 8,
        'ff_hidden_dim': 512
    },
    'N': 10,
    'M': 2
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    "dec_method": "WS",
    'model_load': {
        'path': './POMO/result/train__tsp_n20',
        'info': "MOTSP20 Author's code (Retrain WS Test WS)",
        'epoch': 200,
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 1, #64,
    'aug_batch_size': 200
}
if tester_params['aug_factor'] > 1:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}

# 测试参数
N = 10  # 偏好集大小
BATCH_SIZE = 200  # 测试批次大小

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################
def main():
    """
    主函数，用于测试Geo-CSF模型并生成结果。
    """
    # 根据是否启用数据增强设置不同的文件名后缀
    if tester_params['aug_factor'] == 1:
        sols_floder = f"GeoCSF_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"GeoCSF_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"GeoCSF_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"GeoCSF_hv_n{env_params['problem_size']}.txt"
    else:
        sols_floder = f"GeoCSF(aug)_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"GeoCSF(aug)_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"GeoCSF(aug)_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"GeoCSF(aug)_hv_n{env_params['problem_size']}.txt"

    # 如果处于调试模式，设置调试参数
    if DEBUG_MODE:
        _set_debug_mode()
    
    # 创建日志记录器
    create_logger(**logger_params)
    # 打印配置信息
    _print_config()

    # 记录开始时间
    timer_start = time.time()
    
    # 创建测试器实例
    tester = Tester(env_params=env_params['weca_env_params'],
                    model_params=env_params['weca_model_params'],
                    tester_params=tester_params)
    
    # 复制源代码到结果文件夹
    copy_all_src(tester.result_folder)
    
    # 加载测试数据
    test_path = f"./POMO/data/testdata_tsp_size{env_params['problem_size']}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)

    # 设置参考点用于计算超体积指标
    ref = np.array([20,20])

    batch_size = shared_problem.shape[0]
    
    # 初始化MOCO环境
    moco_env_params = {
        'weca_model_params': env_params['weca_model_params'],
        'weca_env_params': env_params['weca_env_params'],
        'weca_checkpoint_path': env_params['weca_checkpoint_path'],
        'batch_size': BATCH_SIZE,
        'device': 'cuda:' + str(CUDA_DEVICE_NUM) if USE_CUDA else 'cpu',
        'ref_point': env_params['ref_point']
    }
    
    env = MOCOEnv(**moco_env_params)
    
    # 加载训练好的Geo-CSF Actor模型
    actor = MocoPolicyNetwork(**actor_params)
    
    # 加载模型权重
    model_path = f"./result/20251107_143034_train_geo_csf/model_N{N}_best.pth"  # 使用最佳模型
    checkpoint = torch.load(model_path, map_location='cuda:' + str(CUDA_DEVICE_NUM) if USE_CUDA else 'cpu')
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    actor.to('cuda:' + str(CUDA_DEVICE_NUM) if USE_CUDA else 'cpu')
    
    print(f"成功加载Geo-CSF Actor模型: {model_path}")
    
    # 获取 h_graph
    state = env.reset(problem=shared_problem[:BATCH_SIZE])  # 使用测试问题重置环境
    h_graph = state.h_graph  # shape: [B, 128]
    print(f"h_graph shape: {h_graph.shape}")
    
    # 生成偏好集
    with torch.no_grad():
        pref_sets_batch = actor(h_graph)  # shape: [B, N, 2]
    print(f"生成的偏好集 shape: {pref_sets_batch.shape}")
    
    # 初始化存储解的数组
    sols = np.zeros([batch_size, N, 2])
    total_test_time = 0
    
    # 重新循环求解：将偏好分N次喂给求解器
    for i in range(N):
        # 取出第i个偏好向量，shape: [B, 2]
        pref = pref_sets_batch[:, i, :]
        
        # 记录测试开始时间
        test_timer_start = time.time()
        
        # 运行测试
        aug_score = tester.run(shared_problem, pref)
        
        # 记录测试结束时间并累加测试时间
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start
        print('偏好{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

        # 存储结果
        sols[:, i, 0] = np.array(aug_score[0].flatten())
        sols[:, i, 1] = np.array(aug_score[1].flatten())

    # 记录总时间
    timer_end = time.time()
    total_time = timer_end - timer_start

    # 计算并保存最大目标值，用于归一化
    max_obj1 = sols.reshape(-1, 2)[:, 0].max()
    max_obj2 = sols.reshape(-1, 2)[:, 1].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}_geocsf.txt"
    f = open(
        txt2,
        'a')
    f.write(f"MAX OBJ1:{max_obj1}\n")
    f.write(f"MAX OBJ2:{max_obj2}\n")
    f.close()

    
    # 设置单目标TSP的参考解（用于绘图）
    # MOTSP 20
    single_task = [3.83, 3.83]
    
    # 创建并保存帕累托前沿图
    fig = plt.figure()

    sols_mean = sols.mean(0)
    plt.axvline(single_task[0],linewidth=3 , alpha = 0.25)
    plt.axhline(single_task[1],linewidth=3,alpha = 0.25, label = 'Single Objective TSP (Concorde)')
    plt.plot(sols_mean[:,0],sols_mean[:,1], marker = 'o', c = 'C1',ms = 3,  label='Geo-CSF (Ours)')

    plt.legend()
    plt.savefig(F"{tester.result_folder}/{pareto_fig}")

    # 保存平均解
    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f")


    # 计算帕累托解和超体积指标
    nd_sort = Pareto_sols(p_size=env_params['problem_size'], pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)


    # 打印结果统计信息
    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))
    print('NDS: {:.4f}'.format(p_sols_num.float().mean()))
    print('Avg Test Time(s): {:.4f}\n'.format(total_test_time))

    # 保存所有解和超体积指标
    np.savetxt(F"{tester.result_folder}/{all_sols_floder}", sols.reshape(-1, 2),
               delimiter='\t', fmt="%.4f\t%.4f")
    np.savetxt(F"{tester.result_folder}/{hv_floder}", hvs,
               delimiter='\t', fmt="%.4f")

    # 保存测试结果摘要
    if tester_params['aug_factor'] == 1:
        f = open(
            F"{tester.result_folder}/GeoCSF-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"GeoCSF-TSP{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/GeoCSF(aug)-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"GeoCSF(aug)-TSP{env_params['problem_size']}\n")


    f.write(f"MOTSP_2obj Type1\n")
    f.write(f"Model Path: {model_path}\n")
    f.write(f"Hyper Hidden Dim: {env_params['weca_model_params']['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('Run Time(s): {:.4f}\n'.format(total_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]}] \n")
    f.write(f"Info: Geo-CSF测试\n")
    f.close()



##########################################################################################
if __name__ == "__main__":
    main()