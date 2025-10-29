import torch
import numpy as np # 可能需要用于比较原型
import os

# --- 配置区 ---
# 使用相对路径确保文件能被正确找到
data_file = os.path.join(os.path.dirname(__file__), 'conditional_target_data.pt')
num_prototypes = 3 # 您之前设定的原型数量
expected_num_samples = 10000 # 您之前设定的样本总数
expected_h_graph_dim = 128 # 替换为您WE-CA Encoder的embedding_dim
expected_pref_n = 101 # 您设定的偏好集大小
expected_pref_m = 2 # 目标数量
# --- 配置区结束 ---

print(f"Loading data from: {data_file}")
try:
    # 加载数据
    loaded_data = torch.load(data_file)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 开始检查 ---

# 1. 检查整体结构和数量
print("\n--- Basic Checks ---")
if isinstance(loaded_data, list):
    print(f"Data structure is a list: OK")
    print(f"Number of samples: {len(loaded_data)}")
    if len(loaded_data) == expected_num_samples:
        print(f"Number of samples matches expected value ({expected_num_samples}): OK")
    else:
        print(f"Warning: Number of samples ({len(loaded_data)}) does not match expected ({expected_num_samples}).")
else:
    print(f"Error: Expected data structure to be a list, but got {type(loaded_data)}")
    exit()

if not loaded_data:
    print("Error: Loaded data list is empty.")
    exit()

# 2. 检查单个样本的结构、形状和类型
print("\n--- Sample Item Check (Inspecting first item) ---")
first_item = loaded_data[0]
if isinstance(first_item, tuple) and len(first_item) == 2:
    print("Sample structure is a tuple of length 2: OK")
    h_graph, target_prefs = first_item

    # 检查 h_graph
    print("Checking h_graph:")
    if isinstance(h_graph, torch.Tensor):
        print(f"  Type: torch.Tensor - OK")
        print(f"  Shape: {h_graph.shape}")
        if h_graph.shape == (expected_h_graph_dim,):
             print(f"  Shape matches expected ({expected_h_graph_dim},): OK")
        # Handle potential batch dimension if saved differently, e.g., (1, dim)
        elif len(h_graph.shape) == 2 and h_graph.shape[0] == 1 and h_graph.shape[1] == expected_h_graph_dim:
             print(f"  Shape is ({h_graph.shape[0]}, {h_graph.shape[1]}), seems to have batch dim 1 - OK")
        else:
             print(f"  Warning: Shape does not match expected ({expected_h_graph_dim},). Please verify.")
        print(f"  Data type: {h_graph.dtype}")
    else:
        print(f"  Error: Expected h_graph to be a torch.Tensor, but got {type(h_graph)}")

    # 检查 target_prefs
    print("Checking target_prefs:")
    if isinstance(target_prefs, torch.Tensor):
        print(f"  Type: torch.Tensor - OK")
        print(f"  Shape: {target_prefs.shape}")
        if target_prefs.shape == (expected_pref_n, expected_pref_m):
            print(f"  Shape matches expected ({expected_pref_n}, {expected_pref_m}): OK")
        else:
            print(f"  Warning: Shape does not match expected ({expected_pref_n}, {expected_pref_m}). Please verify.")
        print(f"  Data type: {target_prefs.dtype}")
        # 可选：检查数值范围
        # print(f"  Min value: {target_prefs.min()}, Max value: {target_prefs.max()}")
        # print(f"  Sum along last dim (should be approx 1): {target_prefs.sum(dim=-1).mean()}")
    else:
        print(f"  Error: Expected target_prefs to be a torch.Tensor, but got {type(target_prefs)}")

else:
    print(f"Error: Expected sample item to be a tuple of length 2, but got {type(first_item)} with length {len(first_item) if hasattr(first_item, '__len__') else 'N/A'}")

# 3. 检查条件化的核心：目标是否真的不止一种？(抽样检查)
print("\n--- Conditionality Check (Sampling first 100 items) ---")
unique_targets = []
targets_seen_hashes = set()

num_items_to_check = min(100, len(loaded_data))
for i in range(num_items_to_check):
    _, target_prefs_i = loaded_data[i]
    # 使用哈希或者简单的内存地址来快速判断是否见过完全一样的Tensor对象
    target_hash = hash(target_prefs_i.numpy().tobytes()) # 更可靠的判重方式
    if target_hash not in targets_seen_hashes:
         is_new = True
         # Double check equality for safety against hash collisions (rare)
         for existing_target in unique_targets:
             if torch.equal(target_prefs_i, existing_target):
                 is_new = False
                 break
         if is_new:
             unique_targets.append(target_prefs_i)
             targets_seen_hashes.add(target_hash)

print(f"Found {len(unique_targets)} unique target preference sets in the first {num_items_to_check} samples.")
if len(unique_targets) > 1:
    print(f"Conditionality seems present (targets vary): OK")
    if len(unique_targets) == num_prototypes:
         print(f"Number of unique targets matches number of prototypes ({num_prototypes}): Excellent!")
    else:
         print(f"Warning: Found {len(unique_targets)} unique targets, expected {num_prototypes}. Check clustering/assignment.")
elif len(unique_targets) == 1:
    print(f"Warning: Only found 1 unique target set in the sample. Conditional assignment might have failed.")
else: # Should not happen if data is not empty
    print(f"Error: No unique targets found?")


print("\nInspection finished.")