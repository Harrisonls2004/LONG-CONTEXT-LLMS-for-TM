
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class TopicDistQuant(nn.Module):  # 定义主题分布量化类
    '''
        Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder. EMNLP 2020
        基于主题分布量化和负采样解码器的短文本主题建模，EMNLP 2020

        Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao  # 作者信息
    '''
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self._embedding_dim = embedding_dim  # 设置嵌入维度
        self._num_embeddings = num_embeddings  # 设置嵌入数量
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)  # 创建嵌入层
        self._embedding.weight.data.copy_(torch.eye(embedding_dim))  # 用单位矩阵初始化嵌入权重
        self._commitment_cost = commitment_cost  # 设置承诺成本参数

    def forward(self, inputs):  # 定义前向传播方法
        # Calculate distances  # 计算距离
        # NOTE: Do not use torch.cdist. It has unknown bugs.  # 注意：不要使用torch.cdist，它有未知的bug
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)   # 计算输入的平方和
                   + torch.sum(self._embedding.weight**2, dim=1)  # 加上嵌入权重的平方和
                   - 2 * torch.matmul(inputs, self._embedding.weight.t()))  # 减去2倍的内积

        # Encoding  # 编码
        encoding_indices = torch.argmin(distances, dim=1)  # 找到距离最小的嵌入索引

        # Quantize and unflatten  # 量化和展开
        quantized = self._embedding(encoding_indices)  # 根据索引获取量化后的嵌入

        # Loss  # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='none').sum(axis=1).mean()  # 计算编码器潜在损失
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='none').sum(axis=1).mean()  # 计算量化器潜在损失
        loss = q_latent_loss + self._commitment_cost * e_latent_loss  # 计算总损失

        quantized = inputs + (quantized - inputs).detach()  # 使用直通估计器进行梯度传播

        rst = {  # 构建返回结果字典
            'loss': loss,  # 损失
            'quantized': quantized,  # 量化后的结果
            'encoding_indices': encoding_indices,  # 编码索引
        }

        return rst  # 返回结果字典
