##########################################################################################
# 机器环境配置
# DEBUG_MODE: 是否启用调试模式，启用后会使用较小的训练参数以加快调试速度
DEBUG_MODE = False
# USE_CUDA: 是否使用CUDA（GPU），在非调试模式下启用
USE_CUDA = not DEBUG_MODE
# CUDA_DEVICE_NUM: 使用的GPU设备编号
CUDA_DEVICE_NUM = 0

##########################################################################################
# 路径配置
import os
import sys

# 将当前工作目录更改为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 将上级目录添加到Python路径中，以便导入问题定义模块
sys.path.insert(0, "..")  # for problem_def
# 将上上级目录添加到Python路径中，以便导入工具模块
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# 导入必要的模块
import logging
# 从工具模块导入创建日志记录器和复制源代码的函数
from utils.utils import create_logger, copy_all_src

# 从本地导入TSP训练器类
from MOTSPTrainer import TSPTrainer as Trainer

##########################################################################################
# 参数配置

# 环境参数配置
env_params = {
    # 问题规模：TSP城市数量
    'problem_size': 20,
    # POMO种群大小：用于多样化解搜索的样本数量
    'pomo_size': 20,
}

# 模型参数配置
model_params = {
    # 嵌入维度：节点特征向量的维度
    'embedding_dim': 128,
    # 嵌入维度的平方根：用于注意力分数缩放
    'sqrt_embedding_dim': 128**(1/2),
    # 编码器层数：Transformer编码器的层数
    'encoder_layer_num': 6,
    # 查询/键/值维度：注意力机制中QKV向量的维度
    'qkv_dim': 16,
    # 注意力头数：多头注意力机制中的头数
    'head_num': 8,
    # Logits裁剪值：防止数值不稳定，限制logits范围
    'logit_clipping': 10,
    # 前馈网络隐藏层维度：Transformer中前馈网络的隐藏层大小
    'ff_hidden_dim': 512,
    # 评估类型：模型推理时的策略（argmax表示贪婪选择）
    'eval_type': 'argmax',
    # 超网络隐藏层维度：用于生成条件参数的网络维度
    'hyper_hidden_dim': 256,
}

# 优化器参数配置
optimizer_params = {
    # 优化器配置
    'optimizer': {
        # 学习率
        'lr': 1e-4, 
        # 权重衰减（L2正则化）
        'weight_decay': 1e-6
    },
    # 学习率调度器配置
    'scheduler': {
        # 学习率衰减的里程碑epoch
        'milestones': [180,],
        # 衰减因子：学习率乘以0.1
        'gamma': 0.1
    }
}

# 训练器参数配置
trainer_params = {
    # 是否使用CUDA（GPU）
    'use_cuda': USE_CUDA,
    # CUDA设备编号
    'cuda_device_num': CUDA_DEVICE_NUM,
    # 解耦方法：WS可能指加权求和(Weighted Sum)方法处理多目标优化
    'dec_method': 'WS',
    # 训练轮数
    'epochs': 200,
    # 训练episode总数：100,000个训练样本
    'train_episodes': 100 * 1000,
    # 训练批次大小
    'train_batch_size': 64,
    # 日志记录配置
    'logging': {
        # 模型保存间隔（epoch）
        'model_save_interval': 5,
        # 图像保存间隔（epoch）
        'img_save_interval': 10,
        # 图像日志样式参数1
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_20.json'
        },
        # 图像日志样式参数2
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    # 模型加载配置（用于继续训练预训练模型）
    'model_load': {
        # 是否启用模型加载
        'enable': False,
        # 预训练模型和日志文件的目录路径
        'path': './result/saved_tsp20_model',
        # 要加载的预训练模型的epoch版本
        'epoch': 1,
    }
}

# 日志记录器参数配置
logger_params = {
    'log_file': {
        # 日志文件描述
        'desc': 'train__tsp_n20',
        # 日志文件名
        'filename': 'run_log'
    }
}

##########################################################################################
# 主函数
def main():
    # 如果处于调试模式，调整训练参数以加快调试
    if DEBUG_MODE:
        _set_debug_mode()

    # 创建日志记录器
    create_logger(**logger_params)
    # 打印配置信息
    _print_config()

    # 创建训练器实例，传入所有配置参数
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    # 复制所有源代码到结果文件夹（用于实验可重现性）
    copy_all_src(trainer.result_folder)

    # 开始训练
    trainer.run()


# 调试模式参数设置函数
def _set_debug_mode():
    # 声明使用全局变量
    global trainer_params
    # 减少训练轮数以便快速调试
    trainer_params['epochs'] = 2
    # 减少训练样本数以便快速调试
    trainer_params['train_episodes'] = 10
    # 减小批次大小以便快速调试
    trainer_params['train_batch_size'] = 4


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

##########################################################################################

# 程序入口点
if __name__ == "__main__":
    main()