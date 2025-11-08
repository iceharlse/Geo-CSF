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

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

from MOTSPTester import TSPTester as Tester

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
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'hyper_hidden_dim': 256,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    "dec_method": "WS",
    'model_load': {
        'path': './result/train__tsp_n20',
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
def main(n_sols = 101):
    """
    主函数，用于测试MOTSP模型并生成结果。
    
    参数:
    n_sols (int): 用于生成帕累托前沿的解的数量，默认为101。
    """
    # 根据是否启用数据增强设置不同的文件名后缀
    if tester_params['aug_factor'] == 1:
        sols_floder = f"PMOCO_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO_hv_n{env_params['problem_size']}.txt"
    else:
        sols_floder = f"PMOCO(aug)_mean_sols_n{env_params['problem_size']}.txt"
        pareto_fig = f"PMOCO(aug)_Pareto_n{env_params['problem_size']}.png"
        all_sols_floder = f"PMOCO(aug)_all_mean_sols_n{env_params['problem_size']}.txt"
        hv_floder = f"PMOCO(aug)_hv_n{env_params['problem_size']}.txt"

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
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    # 复制源代码到结果文件夹，不知道干啥用的
    copy_all_src(tester.result_folder)
    
    # 加载测试数据
    test_path = f"./data/testdata_tsp_size{env_params['problem_size']}.pt"
    shared_problem = torch.load(test_path).to(device=CUDA_DEVICE_NUM)

    # 设置参考点用于计算超体积指标
    ref = np.array([20,20])    #20

    batch_size = shared_problem.shape[0]
    # 初始化存储解的数组
    sols = np.zeros([batch_size, n_sols, 2])
    total_test_time = 0
    
    # 遍历不同的偏好向量进行测试
    for i in range(n_sols):
        # 创建偏好向量
        pref = torch.zeros(2).cuda()
        pref[0] = 1 - i / (n_sols - 1)
        pref[1] = i / (n_sols - 1)
        pref = pref / torch.sum(pref)

        # 记录测试开始时间
        test_timer_start = time.time()
        # 运行测试
        aug_score = tester.run(shared_problem, pref)
        # 记录测试结束时间并累加测试时间
        test_timer_end = time.time()
        total_test_time += test_timer_end - test_timer_start
        print('Ins{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

        # 存储结果，问题实例数量 x 目标数量（偏好） x 目标值
        sols[:, i, 0] = np.array(aug_score[0].flatten())
        sols[:, i, 1] = np.array(aug_score[1].flatten())

    # 记录总时间
    timer_end = time.time()
    total_time = timer_end - timer_start

    # 计算并保存最大目标值，用于归一化
    max_obj1 = sols.reshape(-1, 2)[:, 0].max()
    max_obj2 = sols.reshape(-1, 2)[:, 1].max()
    txt2 = F"{tester.result_folder}/max_cost_n{env_params['problem_size']}.txt"
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
    plt.plot(sols_mean[:,0],sols_mean[:,1], marker = 'o', c = 'C1',ms = 3,  label='PSL-MOCO (Ours)')

    plt.legend()
    plt.savefig(F"{tester.result_folder}/{pareto_fig}")

    # 保存平均解
    np.savetxt(F"{tester.result_folder}/{sols_floder}", sols_mean,
               delimiter='\t', fmt="%.4f\t%.4f")


    # 计算帕累托解和超体积指标
    nd_sort = Pareto_sols(p_size=env_params['problem_size'], pop_size=sols.shape[0], obj_num=sols.shape[2])  # 用于求解的帕累托解集的实例，输入参数为问题规模，种群规模，目标数量
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t) #求出解集中针对每个问题实例即batch=200的帕累托解集
    p_sols, p_sols_num, _ = nd_sort.show_PE()  # 输出所有问题实例的帕累托解及其数量，p_sols的size为[200, ?, 2]，p_sols_num的size为[200]
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref) #是一个包含200个值的数组，每个值对应一个测试实例的超体积指标


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
            F"{tester.result_folder}/PMOCO-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO-TSP{env_params['problem_size']}\n")
    else:
        f = open(
            F"{tester.result_folder}/PMOCO(aug)-TSP{env_params['pomo_size']}_result.txt",
            'w')
        f.write(f"PMOCO(aug)-TSP{env_params['problem_size']}\n")


    f.write(f"MOTSP_2obj Type1\n")
    f.write(f"Model Path: {tester_params['model_load']['path']}\n")
    f.write(f"Model Epoch: {tester_params['model_load']['epoch']}\n")
    f.write(f"Hyper Hidden Dim: {model_params['hyper_hidden_dim']}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Aug Factor: {tester_params['aug_factor']}\n")
    f.write('Test Time(s): {:.4f}\n'.format(total_test_time))
    f.write('Run Time(s): {:.4f}\n'.format(total_time))
    f.write('HV Ratio: {:.4f}\n'.format(hvs.mean()))
    f.write('NDS: {:.4f}\n'.format(p_sols_num.float().mean()))
    f.write(f"Ref Point:[{ref[0]},{ref[1]}] \n")
    f.write(f"Info: {tester_params['model_load']['info']}\n")
    f.close()



##########################################################################################
if __name__ == "__main__":
    main()