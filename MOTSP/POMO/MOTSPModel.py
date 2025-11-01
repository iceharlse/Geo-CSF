import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):
    """
    多目标旅行商问题(TSP)的神经网络模型
    使用Transformer架构结合POMO(Population-Based Monte Carlo)方法
    """

    def __init__(self, **model_params):
        """
        初始化TSP模型
        
        Args:
            **model_params: 模型参数字典
                - embedding_dim: 嵌入维度
                - encoder_layer_num: 编码器层数
                - head_num: 注意力头数
                - qkv_dim: 查询/键/值维度
                - ff_hidden_dim: 前馈网络隐藏层维度
        """
        super().__init__()
        self.model_params = model_params

        # 编码器：将问题实例编码为节点嵌入
        self.encoder = TSP_Encoder(**model_params)
        # 解码器：基于编码信息逐步选择城市
        self.decoder = TSP_Decoder(**model_params)
        # 存储编码后的节点信息
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state, pref):
        """
        前向传播前的准备工作
        
        Args:
            reset_state: 重置状态，包含问题数据
            pref: 偏好向量，用于多目标优化
        """
        # 对问题数据进行编码
        self.encoded_nodes = self.encoder(reset_state.problems, pref)
        # shape: (batch, problem+1, EMBEDDING_DIM)
        # 为解码器设置键值对缓存
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, route=None, return_probs=False, selected_count=None):
        """
        前向传播函数，决定下一步选择哪个城市
        
        Args:
            state: 当前环境状态
            route: 预定义路线（测试时使用）
            return_probs: 是否返回概率分布
            selected_count: 已选择城市数量
            
        Returns:
            selected: 选择的城市索引
            prob: 选择该城市的概率
            probs: 所有城市的概率分布（可选）
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        # 如果是第一步（还没有选择任何城市）
        if state.current_node is None:
            # 初始化：每个POMO样本从不同起始城市开始
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            # 所有选择的概率初始为1
            prob = torch.ones(size=(batch_size, pomo_size))
            if return_probs:
                probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1))) #这地方有问题后面看看改一下，我觉得除了本体选择的概率是1，其他的应该是0
            # shape: (batch, pomo, problem)

            # 获取第一个节点的编码表示
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            # 为解码器设置第一个查询向量
            self.decoder.set_q1(encoded_first_node)

        # 如果已经选择了至少一个城市
        else:
            # 获取上一个选择节点的编码表示
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            # 使用解码器计算下一个节点的概率分布
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            # 如果没有预定义路线（训练或推理阶段）
            if route is None:
                # 训练模式或softmax评估模式
                if self.training or self.model_params['eval_type'] == 'softmax':
                    # 根据概率分布进行采样选择下一个城市
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    # 获取所选城市的概率
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                # 贪婪评估模式
                else:
                    # 选择概率最高的城市
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    prob = None  # 贪婪选择不需要概率值
            # 如果有预定义路线（验证/测试时）
            else:
                # 按照预定义路线选择城市
                selected = route[:, :, selected_count].reshape(batch_size, pomo_size).long()
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        # 根据参数决定返回内容
        if return_probs:
            return selected, prob, probs
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    根据节点索引从编码节点中提取对应的嵌入表示
    
    Args:
        encoded_nodes: 编码后的所有节点
            shape: (batch, problem, embedding)
        node_index_to_pick: 要提取的节点索引
            shape: (batch, pomo)
            
    Returns:
        picked_nodes: 提取的节点嵌入
            shape: (batch, pomo, embedding)
    """
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    # 构建gather索引
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    # 使用gather操作提取对应节点的嵌入
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# 编码器部分
########################################

class TSP_Encoder(nn.Module):
    """
    TSP问题编码器
    将输入的城市坐标转换为高维嵌入表示
    """

    def __init__(self, **model_params):
        """
        初始化编码器
        
        Args:
            **model_params: 模型参数
        """
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # 偏好向量嵌入层（将偏好向量映射到嵌入空间）
        self.embedding_pref = nn.Linear(2, embedding_dim)
        # 城市坐标嵌入层（将4维坐标映射到嵌入空间）
        self.embedding = nn.Linear(4, embedding_dim)
        # 多层编码器层
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, pref):
        """
        前向传播
        
        Args:
            data: 输入数据（城市坐标）
                shape: (batch, problem, 4)
            pref: 偏好向量
                shape: (batch, 2) 或 (2)
                
        Returns:
            out: 编码后的节点表示
                shape: (batch, problem+1, embedding)
        """
        # data.shape: (batch, problem, 4)

        # 处理 pref 的形状，确保它与批次大小兼容
        if pref.dim() == 1:
            # 如果 pref 是 (2,) 形状，则扩展为 (1, 1, embedding)
            embedded_pref = self.embedding_pref(pref)[None, None, :]  # shape: (1, 1, embedding)
            # 然后扩展到批次大小
            embedded_pref = embedded_pref.repeat(data.shape[0], 1, 1)  # shape: (batch, 1, embedding)
        elif pref.dim() == 2:
            # 如果 pref 是 (batch, 2) 形状，则直接处理
            embedded_pref = self.embedding_pref(pref)  # shape: (batch, embedding)
            embedded_pref = embedded_pref.unsqueeze(1)  # shape: (batch, 1, embedding)
        else:
            raise ValueError(f"pref 的形状不正确: {pref.shape}")

        # 嵌入城市坐标
        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        # 将城市嵌入和偏好嵌入拼接
        out = torch.cat((embedded_input, embedded_pref), dim=1)
        # shape: (batch, problem+1, embedding)
        
        # 逐层处理
        for layer in self.layers:
            out = layer(out)

        return out


class FiLM(nn.Module):
    """
    特征线性调制（Feature-wise Linear Modulation）层
    用于条件特征调整
    """

    def __init__(self, input_size, output_size):
        """
        初始化FiLM层
        
        Args:
            input_size: 条件输入特征大小
            output_size: 待调制特征大小
        """
        super(FiLM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # 生成gamma和beta参数的线性层
        film_output_size = self.output_size * 2
        self.gb_weights = nn.Linear(self.input_size, film_output_size, bias=False)

    def forward(self, x_cond, x_to_film):
        """
        前向传播
        
        Args:
            x_cond: 条件输入
            x_to_film: 待调制的特征
            
        Returns:
            out: 调制后的特征
        """
        # 生成gamma和beta参数
        gb = self.gb_weights(x_cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        # 应用特征调制：out = gamma * x + beta
        out = gamma * x_to_film + beta
        return out


class EncoderLayer(nn.Module):
    """
    编码器层
    包含多头注意力机制和前馈网络
    """

    def __init__(self, **model_params):
        """
        初始化编码器层
        
        Args:
            **model_params: 模型参数
        """
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # 多头注意力机制的线性变换层
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 查询变换
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 键变换
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 值变换
        # 多头注意力输出合并层
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # 残差连接和归一化模块
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        # 前馈网络模块
        self.feedForward = Feed_Forward_Module(**model_params)
        # 残差连接和归一化模块
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        # 节点条件对齐模块（FiLM）
        self.node_conditional_alignment = FiLM(embedding_dim, embedding_dim)

    def forward(self, input1):
        """
        前向传播
        
        Args:
            input1: 输入特征
                shape: (batch, problem+1, EMBEDDING_DIM)
                
        Returns:
            out3: 输出特征
                shape: (batch, problem+1, EMBEDDING_DIM)
        """
        # 复制输入用于后续处理
        input = input1.clone()
        # 使用FiLM对除偏好节点外的所有节点进行条件对齐
        input[:, :-1] = self.node_conditional_alignment(
            x_cond=input1[:, -1, None].repeat(1, input1.shape[1] - 1, 1), 
            x_to_film=input1[:, :-1]
        )

        # input.shape: (batch, problem+1, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        # 计算查询、键、值
        q = reshape_by_heads(self.Wq(input), head_num=head_num)
        k = reshape_by_heads(self.Wk(input), head_num=head_num)
        v = reshape_by_heads(self.Wv(input), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem+1, KEY_DIM)

        # 执行多头注意力计算
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem+1, HEAD_NUM*KEY_DIM)

        # 合并多头注意力输出
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem+1, EMBEDDING_DIM)

        # 第一次残差连接和归一化
        out1 = self.addAndNormalization1(input1, multi_head_out)
        # 前馈网络处理
        out2 = self.feedForward(out1)
        # 第二次残差连接和归一化
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem+1, EMBEDDING_DIM)


########################################
# 解码器部分
########################################

class TSP_Decoder(nn.Module):
    """
    TSP问题解码器
    基于编码信息逐步选择城市构建路径
    """

    def __init__(self, **model_params):
        """
        初始化解码器
        
        Args:
            **model_params: 模型参数
        """
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # 不同类型的查询变换层
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 起始节点查询
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)   # 上一节点查询
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)        # 键变换
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)        # 值变换

        # 多头注意力输出合并层
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # 缓存的键值对，用于多头注意力
        self.k = None  # 保存的键
        self.v = None  # 保存的值
        # 保存的单头注意力键
        self.single_head_key = None
        # 保存的第一个查询向量
        self.q_first = None


    def set_kv(self, encoded_nodes):
        """
        设置键值对缓存
        
        Args:
            encoded_nodes: 编码后的节点
                shape: (batch, problem+1, embedding)
        """
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        # 为多头注意力计算和缓存键值对
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)

        # shape: (batch, head_num, problem+1, qkv_dim)
        
        # 为单头注意力计算和缓存键（排除偏好节点）
        self.single_head_key = encoded_nodes[:, :-1].transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        """
        设置第一个查询向量
        
        Args:
            encoded_q1: 第一个节点的编码
                shape: (batch, n, embedding)  # n可以是1或pomo
        """
        # encoded_q.shape: (batch, n, embedding)
        head_num = self.model_params['head_num']

        # 计算并保存第一个查询向量
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        """
        前向传播，计算下一个城市的选择概率
        
        Args:
            encoded_last_node: 上一个选择节点的编码
                shape: (batch, pomo, embedding)
            ninf_mask: 无效动作掩码（已访问城市标记为-∞）
                shape: (batch, pomo, problem)
                
        Returns:
            probs: 下一个城市的选择概率分布
                shape: (batch, pomo, problem)
        """
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        head_num = self.model_params['head_num']

        # 多头注意力计算
        #######################################################
        # 计算上一个节点的查询向量
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num) #第一次进入decoder的时候，q_last有可能是0向量，后续debug看一下

        # 将第一个查询向量与上一个节点查询向量相加
        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        # 执行多头注意力计算
        out_concat = multi_head_attention(
            q, self.k, self.v, 
            rank3_ninf_mask=torch.cat((ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], 1)), dim=-1) #把3维掩码扩展以包含偏好节点
        )
        # shape: (batch, pomo, head_num*qkv_dim)

        # 合并多头注意力输出
        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        # 单头注意力计算，用于概率计算
        #######################################################
        # 计算注意力得分
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        # 缩放得分
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        # 对数几率裁剪
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # 应用掩码
        score_masked = score_clipped + ninf_mask

        # 计算softmax概率
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# 神经网络子模块和辅助函数
########################################

def reshape_by_heads(qkv, head_num):
    """
    将张量重塑为多头格式
    
    Args:
        qkv: 输入张量
            shape: (batch, n, head_num*key_dim)
        head_num: 注意力头数
            
    Returns:
        q_transposed: 重塑后的张量
            shape: (batch, head_num, n, key_dim)
    """
    # q.shape: (batch, n, head_num*key_dim)

    batch_s = qkv.size(0)
    n = qkv.size(1)

    # 重塑张量以分离多头维度
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    # 转置以获得正确的多头格式
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    多头注意力计算
    
    Args:
        q: 查询向量
            shape: (batch, head_num, n, key_dim) n=pomo
        k: 键向量
            shape: (batch, head_num, problem+1, key_dim)
        v: 值向量
            shape: (batch, head_num, problem+1, key_dim)
        rank2_ninf_mask: 二维掩码
            shape: (batch, problem)
        rank3_ninf_mask: 三维掩码
            shape: (batch, group, problem+1)
            
    Returns:
        out_concat: 注意力输出
            shape: (batch, n, head_num*key_dim)
    """
    # q shape: (batch, head_num, n, key_dim) n在encoder中是problem+1，在decoder中是pomo
    # k,v shape: (batch, head_num, problem+1, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem+1) #这里的problem+1是因为传进来的时候已经包含了偏好节点

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    # 计算注意力得分
    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem+1)

    # 缩放得分
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    # 应用掩码
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # 计算注意力权重
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem+1)

    # 计算加权值
    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    # 转置以恢复正确维度顺序
    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    # 合并多头输出
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    """
    残差连接和归一化模块
    实现Add & Norm结构
    """

    def __init__(self, **model_params):
        """
        初始化模块
        
        Args:
            **model_params: 模型参数
        """
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # 归一化每个特征中所有城市的值
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        """
        前向传播
        
        Args:
            input1: 输入1
                shape: (batch, problem, embedding)
            input2: 输入2
                shape: (batch, problem, embedding)
                
        Returns:
            back_trans: 输出
                shape: (batch, problem, embedding)
        """
        # input.shape: (batch, problem, embedding)

        # 残差连接
        added = input1 + input2
        # shape: (batch, problem, embedding)

        # 转置以适应InstanceNorm1d的要求
        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        # 归一化
        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        # 转置回原始形状
        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    """
    前馈网络模块
    实现位置全连接前馈网络
    """

    def __init__(self, **model_params):
        """
        初始化模块
        
        Args:
            **model_params: 模型参数
        """
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        # 两层线性变换
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        """
        前向传播
        
        Args:
            input1: 输入
                shape: (batch, problem, embedding)
                
        Returns:
            output: 输出
                shape: (batch, problem, embedding)
        """
        # input.shape: (batch, problem, embedding)

        # ReLU激活的两层线性变换
        return self.W2(F.relu(self.W1(input1)))