import torch

def get_random_problems(batch_size, problem_size):
    """
    生成随机的多目标TSP问题实例
    
    参数:
    batch_size (int): 批次大小，即同时生成的问题数量
    problem_size (int): 问题规模，即每个TSP问题包含的城市数量
    
    返回:
    torch.Tensor: 随机生成的问题张量，形状为(batch_size, problem_size, 4)
                 其中最后一个维度的4个值分别表示：
                 - 前两个值(x1, y1)：第一个目标函数（如距离）的城市坐标
                 - 后两个值(x2, y2)：第二个目标函数（如成本）的城市坐标
    """
    # 生成[0,1)范围内的随机数，形状为(batch_size, problem_size, 4)
    # 每个城市都有4个坐标值：两个目标函数各占两个坐标
    problems = torch.rand(size=(batch_size, problem_size, 4))
    return problems

def augment_xy_data_by_64_fold_2obj(xy_data):
    """
    通过对原始数据进行8种几何变换，生成64倍增强的数据
    
    在多目标TSP中，每个目标可能有不同的坐标系统，因此需要分别对两个目标
    的坐标进行变换，然后组合成新的问题实例。
    
    参数:
    xy_data (torch.Tensor): 原始问题数据，形状为(batch_size, problem_size, 4)
                           前两列是第一个目标的坐标(x1, y1)
                           后两列是第二个目标的坐标(x2, y2)
    
    返回:
    torch.Tensor: 增强后的数据，形状为(64*batch_size, problem_size, 4)
    """
   
    # 分离两个目标的坐标数据
    x1 = xy_data[:, :, [0]]  # 第一个目标的x坐标
    y1 = xy_data[:, :, [1]]  # 第一个目标的y坐标
    x2 = xy_data[:, :, [2]]  # 第二个目标的x坐标
    y2 = xy_data[:, :, [3]]  # 第二个目标的y坐标

    # 创建两个字典来存储变换后的数据
    dat1 = {}  # 存储第一个目标的8种变换
    dat2 = {}  # 存储第二个目标的8种变换
    
    # 存储所有增强数据的列表
    dat_aug = []
    
    # 对第一个目标进行8种几何变换：
    # 0: 原始坐标 (x, y)
    dat1[0] = torch.cat((x1, y1), dim=2)
    # 1: 水平翻转 (1-x, y)
    dat1[1] = torch.cat((1-x1, y1), dim=2)
    # 2: 垂直翻转 (x, 1-y)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    # 3: 中心对称 (1-x, 1-y)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    # 4: 沿y=x翻转 (y, x)
    dat1[4] = torch.cat((y1, x1), dim=2)
    # 5: 先沿y=x翻转再水平翻转 (1-y, x)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    # 6: 先沿y=x翻转再垂直翻转 (y, 1-x)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    # 7: 先沿y=x翻转再中心对称 (1-y, 1-x)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)
    
    # 对第二个目标进行相同的8种几何变换：
    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1] = torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4] = torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)
    
    # 组合两个目标的变换结果，生成64种组合（8x8=64）
    for i in range(8):
        for j in range(8):
            # 将第一个目标的第i种变换和第二个目标的第j种变换组合
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)
            
    # 将所有增强后的数据连接成一个张量
    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems