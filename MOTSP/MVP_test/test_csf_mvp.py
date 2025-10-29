import torch
import torch.nn as nn
from torchdiffeq import odeint
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
import random
from sklearn.metrics.pairwise import euclidean_distances
import sys

# 添加上级目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入自定义模块
from csf_module import SetTransformerVectorField
from gfp_module import GeometricFeaturePredictor  # 新增导入
from MOTSPModel import TSPModel as Model  # 导入冻结的WE-CA Encoder

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ODEFuncWrapper(nn.Module):
    """ODE函数包装器"""
    def __init__(self, vector_field, gfp, h_graph):
        super().__init__()
        self.vector_field = vector_field
        self.gfp = gfp  # 新增GFP模块
        self.h_graph = h_graph
    
    def forward(self, t, Lambda_t):
        # vector_field需要t为(B,)形状，而odeint传入的t是标量
        # 我们需要扩展t以匹配批次大小
        batch_size = Lambda_t.shape[0]
        t_batch = t * torch.ones(batch_size, device=Lambda_t.device)
        # 使用GFP生成几何特征
        g_s = self.gfp(self.h_graph)
        return self.vector_field(Lambda_t, t_batch, self.h_graph, g_s)

def load_vector_field_and_gfp(model_path, model_params):
    """加载训练好的vector_field模型和GFP模型"""
    # 创建模型实例
    vector_field = SetTransformerVectorField(**model_params).to(device)
    gfp = GeometricFeaturePredictor(
        input_dim=model_params['condition_dim'],
        hidden_dims=[64, 32],
        output_dim=model_params['geometric_dim']
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载状态字典
    vector_field.load_state_dict(checkpoint['vector_field_state_dict'])
    gfp.load_state_dict(checkpoint['gfp_state_dict'])
    
    vector_field.eval()
    gfp.eval()
    print(f"Vector field and GFP models loaded from: {model_path}")
    return vector_field, gfp

def load_encoder(model_path, model_params):
    """加载冻结的WE-CA Encoder"""
    model = Model(**model_params).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder = model.encoder
    # 冻结encoder参数
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print(f"Encoder loaded and frozen from: {model_path}")
    return encoder

def load_kmeans_model(model_path):
    """加载K-means聚类模型"""
    kmeans_model = load(model_path)
    print(f"K-means model loaded from: {model_path}")
    return kmeans_model

def analyze_cluster_boundaries(h_graphs, kmeans_model, results_dir):
    """分析聚类边界并可视化"""
    # 获取所有h_graphs的numpy数组
    h_graphs_np = np.array([h.cpu().numpy() for h in h_graphs])
    
    # 获取聚类中心
    cluster_centers = kmeans_model.cluster_centers_
    n_clusters = len(cluster_centers)
    
    # 计算每个点到所有聚类中心的距离
    distances_to_centers = euclidean_distances(h_graphs_np, cluster_centers)
    
    # 预测每个点的聚类标签
    predicted_labels = kmeans_model.predict(h_graphs_np)
    
    # 创建聚类可视化图
    plt.figure(figsize=(15, 5))
    
    # 绘制聚类结果
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green']
    cluster_names = ['Uniform', 'Focus Obj1', 'Focus Obj2']
    
    for i in range(n_clusters):
        mask = predicted_labels == i
        plt.scatter(h_graphs_np[mask, 0], h_graphs_np[mask, 1], 
                   c=colors[i], label=f'{cluster_names[i]} (Cluster {i})', alpha=0.7)
    
    # 绘制聚类中心
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Cluster Centers')
    plt.xlabel('h_graph dimension 1')
    plt.ylabel('h_graph dimension 2')
    plt.title('Clustering of Test Data')
    plt.legend()
    plt.grid(True)
    
    # 绘制到聚类中心的距离热图
    plt.subplot(1, 3, 2)
    im = plt.imshow(distances_to_centers, aspect='auto', cmap='viridis')
    plt.xlabel('Cluster Centers')
    plt.ylabel('Test Samples')
    plt.title('Distance to Cluster Centers')
    plt.colorbar(im, label='Euclidean Distance')
    plt.xticks(range(n_clusters), [f'Center {i}' for i in range(n_clusters)])
    
    # 绘制边界分析图
    plt.subplot(1, 3, 3)
    # 计算每个点到其分配聚类中心的距离
    assigned_center_distances = distances_to_centers[np.arange(len(predicted_labels)), predicted_labels]
    
    # 计算每个点到其他聚类中心的最小距离
    other_center_distances = []
    for i in range(len(h_graphs_np)):
        other_distances = [distances_to_centers[i, j] for j in range(n_clusters) if j != predicted_labels[i]]
        other_center_distances.append(min(other_distances) if other_distances else float('inf'))
    other_center_distances = np.array(other_center_distances)
    
    # 计算边界指标（到其他中心的距离与到分配中心距离的比值）
    boundary_ratio = other_center_distances / (assigned_center_distances + 1e-8)  # 避免除零
    
    # 绘制散点图，颜色表示边界指标
    scatter = plt.scatter(assigned_center_distances, other_center_distances, 
                         c=boundary_ratio, cmap='plasma', alpha=0.7)
    plt.xlabel('Distance to Assigned Cluster Center')
    plt.ylabel('Distance to Nearest Other Cluster Center')
    plt.title('Boundary Analysis\n(Color indicates boundary ratio)')
    plt.plot([0, max(assigned_center_distances)], [0, max(assigned_center_distances)], 
             'r--', alpha=0.5, label='Equal Distance Line')
    plt.legend()
    plt.colorbar(scatter, label='Boundary Ratio')
    plt.grid(True)
    
    # 保存聚类分析图
    cluster_analysis_path = os.path.join(results_dir, "cluster_analysis.png")
    plt.tight_layout()
    plt.savefig(cluster_analysis_path)
    plt.close()
    print(f"Cluster analysis plot saved to: {cluster_analysis_path}")
    
    # 找出边界区域的样本（边界比率接近1的样本）
    boundary_threshold = 0.8  # 可调整的阈值
    boundary_indices = np.where(boundary_ratio > boundary_threshold)[0]
    
    print(f"\nBoundary Analysis Summary:")
    print(f"- Total test samples analyzed: {len(h_graphs_np)}")
    print(f"- Number of boundary samples (ratio > {boundary_threshold}): {len(boundary_indices)}")
    print(f"- Average boundary ratio: {np.mean(boundary_ratio):.4f}")
    print(f"- Max boundary ratio: {np.max(boundary_ratio):.4f}")
    
    return predicted_labels, distances_to_centers, boundary_indices

def main():
    # 模型参数（与训练时保持一致）
    model_params = {
        'input_dim': 2,
        'hidden_dim': 128,
        'condition_dim': 128,
        'geometric_dim': 16,  # 新增几何特征维度参数
        'num_layers': 2,
        'num_heads': 8,
        'ff_hidden_dim': 512,
        'time_embed_dim': 128
    }
    
    # 加载训练好的vector_field模型和GFP模型
    vector_field_path = os.path.join(os.path.dirname(__file__), "checkpoints", "csf_mvp_best.pt")
    vector_field, gfp = load_vector_field_and_gfp(vector_field_path, model_params)
    
    # 加载冻结的WE-CA Encoder（如果需要重新计算h_graph）
    # 注意：在test_conditional_target_data.pt中已经保存了h_graph，所以这里可能不需要重新加载Encoder
    # 但为了完整性，我们仍然提供加载方法
    # encoder_path = os.path.join(os.path.dirname(__file__), "result", "train__tsp_n20", "checkpoint_motsp-200.pt")
    # encoder = load_encoder(encoder_path, model_params)
    
    # 加载K-means聚类模型
    kmeans_model_path = os.path.join(os.path.dirname(__file__), "kmeans_model.joblib")
    kmeans_model = load_kmeans_model(kmeans_model_path)
    
    # 加载测试数据
    test_data_path = os.path.join(os.path.dirname(__file__), "test_conditional_target_data.pt")
    test_data = torch.load(test_data_path, map_location='cpu')  # 先加载到CPU，使用时再移动到设备
    print(f"Loaded {len(test_data)} test samples from: {test_data_path}")
    
    # 创建保存结果的目录
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 随机抽取100个测试样本（如果测试集少于100个，则使用全部样本）
    num_test_samples = min(100, len(test_data))
    # 生成随机索引
    random_indices = random.sample(range(len(test_data)), num_test_samples)
    print(f"Testing on {num_test_samples} randomly selected samples...")
    
    # 收集用于聚类分析的h_graphs
    h_graphs_for_analysis = []
    for idx in random_indices:
        h_graph, _ = test_data[idx]
        h_graphs_for_analysis.append(h_graph)
    
    # 分析聚类边界
    predicted_labels, distances_to_centers, boundary_indices = analyze_cluster_boundaries(
        h_graphs_for_analysis, kmeans_model, results_dir)
    
    # 生成与验证循环
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            # 获取测试数据
            h_graph, target_prefs_k = test_data[idx]
            h_graph = h_graph.to(device).unsqueeze(0)  # 添加批次维度
            target_prefs_k = target_prefs_k.to(device)
            
            print(f"\nProcessing sample {i+1}/{num_test_samples} (original index: {idx})")
            print(f"  h_graph shape: {h_graph.shape}")
            print(f"  target_prefs_k shape: {target_prefs_k.shape}")
            
            # 获取维度
            B, N, M = 1, target_prefs_k.shape[0], target_prefs_k.shape[1]
            
            # 采样初始噪声
            Lambda_0 = torch.randn(B, N, M).to(device)
            print(f"  Lambda_0 shape: {Lambda_0.shape}")
            
            # 定义时间点
            t_span = torch.tensor([0.0, 1.0]).to(device)
            
            # 创建ODE函数包装器
            current_ode_func = ODEFuncWrapper(vector_field, gfp, h_graph)
            
            # 求解ODE
            print("  Solving ODE...")
            Lambda_1_solution = odeint(
                current_ode_func, 
                Lambda_0, 
                t_span, 
                method='rk4', 
                options={'step_size': 0.01}  # 使用更小的步长以获得更精确的结果
            )
            
            # 取t=1时的结果
            Lambda_1 = Lambda_1_solution[-1]
            print(f"  Lambda_1 shape: {Lambda_1.shape}")
            
            # 检查是否为边界样本
            is_boundary_sample = i in boundary_indices
            if is_boundary_sample:
                print(f"  *** BOUNDARY SAMPLE DETECTED ***")
                assigned_distance = distances_to_centers[i, predicted_labels[i]]
                # 计算到其他聚类中心的最小距离
                other_distances = [distances_to_centers[i, j] for j in range(len(kmeans_model.cluster_centers_)) if j != predicted_labels[i]]
                nearest_other_distance = min(other_distances) if other_distances else float('inf')
                print(f"  Distance to assigned center: {assigned_distance:.4f}")
                print(f"  Distance to nearest other center: {nearest_other_distance:.4f}")
                print(f"  Boundary ratio: {nearest_other_distance/(assigned_distance + 1e-8):.4f}")
            
            # 评估与可视化
            # 将生成的Λ_1.squeeze(0)转换成numpy
            Lambda_1_np = Lambda_1.squeeze(0).cpu().numpy()
            target_prefs_k_np = target_prefs_k.cpu().numpy()
            
            # 绘制散点图对比
            plt.figure(figsize=(15, 5))
            
            # 绘制生成的Λ_1
            plt.subplot(1, 3, 1)
            plt.scatter(Lambda_1_np[:, 0], Lambda_1_np[:, 1], c='blue', label='Generated Λ₁', alpha=0.7)
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title(f'Generated Preferences (Sample {i+1})')
            plt.grid(True)
            plt.legend()
            
            # 绘制目标原型Λ*_k
            plt.subplot(1, 3, 2)
            plt.scatter(target_prefs_k_np[:, 0], target_prefs_k_np[:, 1], c='red', label='Target Λ*_k', alpha=0.7)
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title(f'Target Preferences (Sample {i+1})')
            plt.grid(True)
            plt.legend()
            
            # 绘制h_graph在聚类空间中的位置
            plt.subplot(1, 3, 3)
            h_graph_np = h_graphs_for_analysis[i].numpy()
            colors = ['red', 'blue', 'green']
            cluster_names = ['Uniform', 'Focus Obj1', 'Focus Obj2']
            
            # 绘制所有聚类中心
            for j, center in enumerate(kmeans_model.cluster_centers_):
                plt.scatter(center[0], center[1], c=colors[j], marker='x', s=200, 
                           label=f'{cluster_names[j]} Center', alpha=0.7)
            
            # 绘制当前样本点
            color = 'orange' if is_boundary_sample else 'purple'
            label = 'Current Sample (Boundary)' if is_boundary_sample else 'Current Sample'
            plt.scatter(h_graph_np[0], h_graph_np[1], c=color, s=100, marker='o', 
                       label=label, edgecolors='black', linewidth=1)
            
            plt.xlabel('h_graph dimension 1')
            plt.ylabel('h_graph dimension 2')
            plt.title(f'h_graph Position in Cluster Space')
            plt.grid(True)
            plt.legend()
            
            # 保存图像
            comparison_path = os.path.join(results_dir, f"comparison_sample_{i+1}_idx_{idx}.png")
            plt.tight_layout()
            plt.savefig(comparison_path)
            plt.close()
            print(f"  Comparison plot saved to: {comparison_path}")
            
            # 计算生成结果与目标原型之间的距离
            diff = np.linalg.norm(Lambda_1_np - target_prefs_k_np, axis=1)
            mean_diff = np.mean(diff)
            print(f"  Mean distance between generated and target: {mean_diff:.4f}")
    
    print(f"\nTesting completed. Results saved to: {results_dir}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    main()