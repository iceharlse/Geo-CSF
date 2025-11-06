import logging
import os
import sys
import torch

# 将当前工作目录更改为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# 将上级目录添加到Python路径中，以便导入问题定义模块和工具模块
parent_dir = os.path.join(script_dir, "..")
sys.path.insert(0, parent_dir)

# --- 导入---
from utils.utils import create_logger, copy_all_src
from Trainer import GeoCSFTrainer as Trainer   
from MOTSP.moco_env import MOCOEnv

# --- 机器/CUDA 设置---
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0 # 假设您使用 GPU 0

##########################################################################################
# 参数配置
##########################################################################################

# --- 环境参数---
env_params = {
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
        'pomo_size': 20,
    },
    'weca_checkpoint_path': './POMO/result/train__tsp_n20/checkpoint_motsp-200.pt',
    'ref_point': [20.0, 20.0]
}

# --- Actor 参数---
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
    'M': 2
}

# --- Critic 参数---
critic_params = {
    'condition_dim': 128,
    'M': 2,
    'hidden_dim': 128,
}

# --- 优化器参数 ---
optimizer_params = {
    'lr_actor': 1e-4,
    'lr_critic': 1e-4
}

# --- 训练器参数---
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'device': torch.device('cuda:0' if USE_CUDA else 'cpu'),
    
    # 课程学习
    'curriculum_stages': [
        (10, 1000),   # 阶段 1: N=10, 跑 1000 个 episodes
        (20, 2000),   # 阶段 2: N=20, 跑 2000 个 episodes
        (50, 3000),   # 阶段 3: N=50, 跑 3000 个 episodes
        (101, 5000)   # 阶段 4: N=101, 跑 5000 个 episodes
    ],
    
    # Actor-Critic 参数
    'batch_size': 32,
    'buffer_capacity': 100000,
    'gamma': 0.99,
    'tau': 0.005,
    'noise_level': 0.1,
    'train_steps_per_episode': 1, # 每个 episode 训练几次
    'start_train_after_episodes': 100, # 收集多少数据后才开始训练
    
    # 日志和保存
    'logging': {
        'model_save_interval': 5, # 每 5 个 stage 保存一次模型
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,
        'path': './result/saved_geo_csf_model',
        'stage_to_load': 0, # 从第 0 阶段的 checkpoint 开始
    }
}

# --- 日志参数 ---
logger_params = {
    'log_file': {
        'desc': 'train_geo_csf',
        'filename': 'run_log'
    }
}

##########################################################################################
# 主函数
##########################################################################################

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # 1. 创建环境
    # 构建完整的 moco_env_params 字典
    moco_env_params = {
        'weca_model_params': env_params['weca_model_params'],
        'weca_env_params': env_params['weca_env_params'],
        'weca_checkpoint_path': env_params['weca_checkpoint_path'],
        'batch_size': trainer_params['batch_size'],
        'device': trainer_params['device'],
        'ref_point': env_params['ref_point']
    }
    
    env = MOCOEnv(**moco_env_params)

    # 2. 创建训练器
    trainer = Trainer(env=env, # <-- 传入实例
                      actor_params=actor_params,
                      critic_params=critic_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    
    copy_all_src(trainer.result_folder)
    
    # 3. 开始训练
    trainer.run()

# 调试模式参数设置函数
def _set_debug_mode():
    # 声明使用全局变量
    global trainer_params
    # 修改课程学习阶段以减少训练时间
    trainer_params['curriculum_stages'] = [
        (10, 10),   # 阶段 1: N=10, 跑 10 个 episodes
        (20, 10),   # 阶段 2: N=20, 跑 10 个 episodes
    ]
    # 减小缓冲区容量以便快速调试
    trainer_params['buffer_capacity'] = 1000
    # 减少开始训练前需要收集的数据量
    trainer_params['start_train_after_episodes'] = 5


# 打印配置信息函数
def _print_config():
    # 获取根日志记录器
    logger = logging.getLogger('root')
    # 记录调试模式状态
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    # 记录CUDA使用情况和设备编号
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    # 记录所有以'params'结尾的全局变量（即各种配置参数）
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    main()