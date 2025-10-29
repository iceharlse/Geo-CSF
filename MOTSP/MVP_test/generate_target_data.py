import torch
import numpy as np
from sklearn.cluster import KMeans
import os
import sys
from joblib import dump  # 使用joblib替代pickle

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入必要的模块
from MOTSProblemDef import get_random_problems
from MOTSPModel import TSPModel

def generate_prototype_preferences(N=101, M=2):
    """
    生成三类原型目标偏好集
    
    Args:
        N (int): 偏好向量数量
        M (int): 目标维度
        
    Returns:
        dict: 包含三类原型偏好向量的字典
    """
    # prototype_A: 标准的线性插值向量 (均匀分布)
    prototype_A = np.zeros((N, M))
    for i in range(N):
        prototype_A[i, 0] = 1 - i / (N - 1)
        prototype_A[i, 1] = i / (N - 1)
    
    # prototype_B: 向量密集分布在 (1, 0) 附近 (偏重目标1)
    prototype_B = np.zeros((N, M))
    # 使用非线性分布，使更多向量集中在(1,0)附近
    for i in range(N):
        t = i / (N - 1)  # t从0到1
        # 使用平方函数使分布偏向0端
        prototype_B[i, 0] = 1 - t**2
        prototype_B[i, 1] = t**2
    
    # prototype_C: 向量密集分布在 (0, 1) 附近 (偏重目标2)
    prototype_C = np.zeros((N, M))
    # 使用非线性分布，使更多向量集中在(0,1)附近
    for i in range(N):
        t = i / (N - 1)  # t从0到1
        # 使用平方函数使分布偏向1端
        prototype_C[i, 0] = (1-t)**2
        prototype_C[i, 1] = 1 - (1-t)**2
    
    return {
        'uniform': torch.FloatTensor(prototype_A),
        'focus_objective1': torch.FloatTensor(prototype_B),
        'focus_objective2': torch.FloatTensor(prototype_C)
    }

def load_pretrained_encoder(model_path, model_params, device):
    """
    加载预训练模型并提取冻结的Encoder
    
    Args:
        model_path (str): 预训练模型路径
        model_params (dict): 模型参数
        device: 设备(CUDA/CPU)
        
    Returns:
        nn.Module: 冻结的Encoder
    """
    # 实例化模型
    model = TSPModel(**model_params)
    model = model.to(device)
    
    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 提取Encoder并冻结参数
    encoder = model.encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    return encoder

def compute_h_graph(encoder, problems, pref, device):
    """
    使用冻结的Encoder计算h_graph
    
    Args:
        encoder: 冻结的Encoder
        problems: 问题实例
        pref: 偏好向量
        device: 设备
        
    Returns:
        torch.Tensor: h_graph表示
    """
    with torch.no_grad():
        # 确保输入在正确的设备上
        problems = problems.to(device)
        pref = pref.to(device)
        
        # 计算h_graph
        h_graph = encoder(problems, pref)
        # 只取城市节点的嵌入，不包括偏好节点（最后一个节点）
        h_graph = h_graph[:, :-1, :]  # shape: (batch, problem, embedding)
        
        # 对所有城市节点的嵌入进行平均，得到图级别的表示
        h_graph = torch.mean(h_graph, dim=1)  # shape: (batch, embedding)
        
    return h_graph

def generate_test_dataset(encoder, kmeans_model, prototypes, env_params, device, num_test_samples=1000):
    """
    生成测试数据集
    
    Args:
        encoder: 冻结的Encoder
        kmeans_model: 训练好的K-means模型
        prototypes: 原型偏好向量
        env_params: 环境参数
        device: 设备
        num_test_samples: 测试样本数量
        
    Returns:
        list: 测试数据集
    """
    print("Generating test dataset...")
    test_data = []
    batch_size = 100
    num_batches = num_test_samples // batch_size
    uniform_pref = torch.FloatTensor([0.5, 0.5]).to(device)
    
    for i in range(num_batches):
        # 生成随机问题实例
        problems = get_random_problems(batch_size, env_params['problem_size'])
        
        # 计算h_graph
        h_graph = compute_h_graph(encoder, problems, uniform_pref, device)
        
        # 预测每个h_graph属于哪个类别
        cluster_labels = kmeans_model.predict(h_graph.cpu().numpy())
        
        # 根据类别标签获取对应的目标原型
        for j in range(batch_size):
            label = cluster_labels[j]
            # 根据标签选择对应的原型集
            if label == 0:
                target_prefs = prototypes['uniform']
            elif label == 1:
                target_prefs = prototypes['focus_objective1']
            else:  # label == 2
                target_prefs = prototypes['focus_objective2']
            
            # 添加数据对到列表中
            test_data.append((h_graph[j].cpu(), target_prefs))
        
        if (i + 1) % 10 == 0:
            print(f"Generated {min((i + 1) * batch_size, num_test_samples)} test samples")
    
    print(f"Test dataset generation completed. Total samples: {len(test_data)}")
    return test_data

def main():
    # 设置设备
    device = torch.device('cuda:1')
    print(f"Using device: {device}")
    
    # 模型参数（根据训练脚本中的配置）
    model_params = {
        'embedding_dim': 128,
        'encoder_layer_num': 6,
        'head_num': 8,
        'qkv_dim': 16,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'hyper_hidden_dim': 256,
    }
    
    # 环境参数
    env_params = {
        'problem_size': 20,  # 可根据需要修改
        'pomo_size': 20,
    }
    
    # 使用绝对路径设置预训练模型路径
    model_path = os.path.join(os.path.dirname(__file__), "result", "train__tsp_n20", "checkpoint_motsp-200.pt")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # 加载并冻结Encoder
    print("Loading pretrained encoder...")
    encoder = load_pretrained_encoder(model_path, model_params, device)
    print("Encoder loaded and frozen.")
    
    # 定义原型目标偏好集
    print("Generating prototype preferences...")
    prototypes = generate_prototype_preferences(N=101, M=2)
    print(f"Generated {len(prototypes)} prototype sets.")
    
    # 设置聚类参数
    K = len(prototypes)  # 聚类数量等于原型数量
    num_total_samples = 10000  # 总样本数
    batch_size = 100  # 每批处理的样本数
    
    # 生成大量问题实例并计算h_graph用于聚类
    print("Generating problems and computing h_graph for clustering...")
    all_h_graphs = []
    
    # 使用均匀偏好向量[0.5, 0.5]来计算h_graph用于聚类
    uniform_pref = torch.FloatTensor([0.5, 0.5]).to(device)
    
    num_batches_for_clustering = 100  # 用于聚类的批次数量 num_total_samples // batch_size
    for i in range(num_batches_for_clustering):
        # 生成随机问题实例
        problems = get_random_problems(batch_size, env_params['problem_size'])
        
        # 计算h_graph
        h_graph = compute_h_graph(encoder, problems, uniform_pref, device) # shape: (batch_size, embedding)
        all_h_graphs.append(h_graph.cpu())  # 移到CPU以节省GPU内存
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_batches_for_clustering} batches for clustering")
    
    # 合并所有h_graph
    all_h_graphs = torch.cat(all_h_graphs, dim=0)  # shape: (num_batches_for_clustering * batch_size, embedding)
    print(f"Total h_graphs for clustering: {all_h_graphs.shape[0]}")
    
    # 训练K-Means模型
    print("Training K-Means model...")
    kmeans_model = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans_model.fit(all_h_graphs.numpy())
    print("K-Means training completed.")
    
    # 保存K-means模型参数（使用joblib）
    kmeans_model_path = os.path.join(os.path.dirname(__file__), 'kmeans_model.joblib')
    dump(kmeans_model, kmeans_model_path)
    print(f"K-Means model saved to {kmeans_model_path}")
    
    # 生成最终的监督学习数据对
    print("Generating supervised learning data pairs...")
    supervised_data = []
    
    # 重新生成数据用于训练
    num_generation_batches = num_total_samples // batch_size
    for i in range(num_generation_batches):
        # 生成随机问题实例
        problems = get_random_problems(batch_size, env_params['problem_size'])
        
        # 计算h_graph
        h_graph = compute_h_graph(encoder, problems, uniform_pref, device)
        
        # 预测每个h_graph属于哪个类别
        cluster_labels = kmeans_model.predict(h_graph.cpu().numpy())
        
        # 根据类别标签获取对应的目标原型
        for j in range(batch_size):
            label = cluster_labels[j]
            # 根据标签选择对应的原型集
            if label == 0:
                target_prefs = prototypes['uniform']
            elif label == 1:
                target_prefs = prototypes['focus_objective1']
            else:  # label == 2
                target_prefs = prototypes['focus_objective2']
            
            # 添加数据对到列表中
            supervised_data.append((h_graph[j].cpu(), target_prefs))
        
        if (i + 1) % 10 == 0:
            print(f"Generated {min((i + 1) * batch_size, num_total_samples)}/{num_total_samples} data pairs")
    
    # 保存训练数据
    output_file = 'conditional_target_data.pt'
    torch.save(supervised_data, output_file)
    print(f"Saved {len(supervised_data)} training data pairs to {output_file}")
    
    # 生成测试数据集
    test_data = generate_test_dataset(encoder, kmeans_model, prototypes, env_params, device, num_test_samples=1000)
    
    # 保存测试数据
    test_output_file = 'test_conditional_target_data.pt'
    torch.save(test_data, test_output_file)
    print(f"Saved {len(test_data)} test data pairs to {test_output_file}")
    
    # 打印一些统计信息
    print("\nData generation summary:")
    print(f"- Generated {len(supervised_data)} supervised training data pairs")
    print(f"- Generated {len(test_data)} test data pairs")
    print(f"- Each pair contains:")
    print(f"  * h_graph: Graph representation tensor")
    print(f"  * target_prefs: Target preference vectors matched by cluster")
    print(f"- Prototype sets:")
    for i, (key, prefs) in enumerate(prototypes.items()):
        print(f"  * {key}: {prefs.shape[0]} preference vectors")
    print(f"- K-Means model saved to: {kmeans_model_path}")

if __name__ == "__main__":
    main()