import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import logging
from tqdm import tqdm
import math

# 导入自定义模块
from csf_module import SetTransformerVectorField
from gfp_module import GeometricFeaturePredictor  # 新增导入

# 自定义数据集类
class ConditionalFlowMatchingDataset(Dataset):
    """条件流匹配数据集"""
    def __init__(self, data_list):
        """
        Args:
            data_list: 包含(h_graph, target_prefs)元组的列表
        """
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        h_graph, target_prefs = self.data_list[idx]
        return h_graph, target_prefs

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练条件流匹配向量场模型')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                        default='conditional_target_data.pt',
                        help='条件化监督数据文件路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='验证集比例')
    
    # 模型超参数
    parser.add_argument('--input_dim', type=int, default=2,
                        help='输入/输出偏好向量的维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Set Transformer内部的主要隐藏维度')
    parser.add_argument('--condition_dim', type=int, default=128,
                        help='条件向量维度')
    parser.add_argument('--geometric_dim', type=int, default=16,  # 新增几何特征维度参数
                        help='几何特征维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Set Transformer块的数量')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='多头注意力机制的头数')
    parser.add_argument('--ff_hidden_dim', type=int, default=512,
                        help='FFN内部隐藏层维度')
    parser.add_argument('--time_embed_dim', type=int, default=128,
                        help='时间嵌入维度')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='优化器学习率')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='训练的总轮数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备 (cuda 或 cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='保存模型检查点的目录')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help='时间采样epsilon修正值')
    
    return parser.parse_args()

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def prepare_data(args):
    """准备训练和验证数据"""
    logger = logging.getLogger(__name__)
    
    # 加载原始数据
    logger.info(f"加载数据从: {args.data_path}")
    loaded_data = torch.load(args.data_path)
    logger.info(f"数据加载完成，共 {len(loaded_data)} 个样本")
    
    # 划分训练集和验证集
    total_samples = len(loaded_data)
    val_samples = int(total_samples * args.validation_split)
    train_samples = total_samples - val_samples
    
    logger.info(f"训练样本数: {train_samples}, 验证样本数: {val_samples}")
    
    # 创建数据集
    train_dataset = ConditionalFlowMatchingDataset(loaded_data[:train_samples])
    val_dataset = ConditionalFlowMatchingDataset(loaded_data[train_samples:])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    """主训练函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始条件流匹配模型训练")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 准备数据
    train_loader, val_loader = prepare_data(args)
    
    # 实例化模型
    model_params = {
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'condition_dim': args.condition_dim,
        'geometric_dim': args.geometric_dim,  # 新增几何特征维度参数
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'ff_hidden_dim': args.ff_hidden_dim,
        'time_embed_dim': args.time_embed_dim
    }
    
    logger.info("创建模型...")
    vector_field = SetTransformerVectorField(**model_params).to(device)
    logger.info(f"模型参数数量: {sum(p.numel() for p in vector_field.parameters())}")
    
    # 实例化几何特征预测器
    gfp = GeometricFeaturePredictor(
        input_dim=args.condition_dim,
        hidden_dims=[64, 32],
        output_dim=args.geometric_dim
    ).to(device)
    logger.info(f"GFP模型参数数量: {sum(p.numel() for p in gfp.parameters())}")
    
    # 设置优化器（同时优化vector_field和gfp的参数）
    optimizer = optim.Adam(
        list(vector_field.parameters()) + list(gfp.parameters()), 
        lr=args.learning_rate
    )
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 训练阶段
        vector_field.train()
        gfp.train()
        train_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for h_graph_batch, target_prefs_batch in progress_bar:
            num_batches += 1
            
            # 将数据移到设备
            h_graph_batch = h_graph_batch.to(device)
            target_prefs_batch = target_prefs_batch.to(device)
            
            B, N, M = target_prefs_batch.shape
            
            # 生成几何特征 g_s
            g_s_batch = gfp(h_graph_batch)
            
            # 核心CFM训练逻辑
            # 1. 采样噪声 Λ_0
            Lambda_0 = torch.randn_like(target_prefs_batch)
            
            # 2. 采样时间 t (使用epsilon修正防止数值问题)
            # t 的采样范围是 [epsilon, 1-epsilon] 而不是 [0, 1]
            t = torch.rand(B, 1, 1, device=device) * (1 - 2 * args.epsilon) + args.epsilon
            
            # 3. 线性插值 Λ_t
            Lambda_t = (1 - t) * Lambda_0 + t * target_prefs_batch
            
            # 4. 计算目标速度 u_t
            u_t = target_prefs_batch - Lambda_0
            
            # 5. 模型前向传播
            # 注意 t 需要 squeeze 以匹配模型 forward 的输入要求
            predicted_v = vector_field(Lambda_t, t.squeeze(-1).squeeze(-1), h_graph_batch, g_s_batch)
            
            # 6. 计算损失 (显式设置reduction参数)
            loss = F.mse_loss(predicted_v, u_t, reduction='mean')
            
            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(vector_field.parameters()) + list(gfp.parameters()), args.gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        avg_train_loss = train_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 平均训练损失: {avg_train_loss:.6f}")
        
        # 验证阶段
        vector_field.eval()
        gfp.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for h_graph_batch, target_prefs_batch in tqdm(val_loader, desc="验证"):
                val_batches += 1
                
                # 将数据移到设备
                h_graph_batch = h_graph_batch.to(device)
                target_prefs_batch = target_prefs_batch.to(device)
                
                B, N, M = target_prefs_batch.shape
                
                # 生成几何特征 g_s
                g_s_batch = gfp(h_graph_batch)
                
                # 执行与训练时相同的CFM逻辑
                # 1. 采样噪声 Λ_0
                Lambda_0 = torch.randn_like(target_prefs_batch)
                
                # 2. 采样时间 t (使用epsilon修正防止数值问题)
                t = torch.rand(B, 1, 1, device=device) * (1 - 2 * args.epsilon) + args.epsilon
                
                # 3. 线性插值 Λ_t
                Lambda_t = (1 - t) * Lambda_0 + t * target_prefs_batch
                
                # 4. 计算目标速度 u_t
                u_t = target_prefs_batch - Lambda_0
                
                # 5. 模型前向传播
                predicted_v = vector_field(Lambda_t, t.squeeze(-1).squeeze(-1), h_graph_batch, g_s_batch)
                
                # 6. 计算损失 (显式设置reduction参数)
                loss = F.mse_loss(predicted_v, u_t, reduction='mean')
                
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_batches
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 平均验证损失: {avg_val_loss:.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（包括vector_field和gfp）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.save_dir, "csf_mvp_best.pt")
            # 保存两个模型的状态字典
            torch.save({
                'vector_field_state_dict': vector_field.state_dict(),
                'gfp_state_dict': gfp.state_dict()
            }, best_model_path)
            logger.info(f"最佳模型已保存到: {best_model_path}")
        
        # 定期保存模型检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"csf_mvp_epoch_{epoch+1}.pt")
            torch.save({
                'vector_field_state_dict': vector_field.state_dict(),
                'gfp_state_dict': gfp.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            logger.info(f"模型检查点已保存到: {checkpoint_path}")
    
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()