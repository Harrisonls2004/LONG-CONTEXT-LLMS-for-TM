import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
from . import utils  # 导入工具模块


class CDDecoder(nn.Module):  # 定义层次化解码器类
    def __init__(self, num_layers, vocab_size, bias_p, bias_topk):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.num_layers = num_layers  # 设置层数
        self.bias_p = bias_p  # 设置偏置参数
        self.bias_topk = bias_topk  # 设置偏置top-k参数

        self.decoder_bn_list = nn.ModuleList([nn.BatchNorm1d(vocab_size, affine=False) for _ in range(num_layers)])  # 创建解码器批归一化层列表

        self.bias_vectors = nn.ParameterList([])  # 初始化偏置向量参数列表
        for _ in range(num_layers):  # 遍历每一层
            bias_vector = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, vocab_size)))  # 创建Xavier初始化的偏置向量
            self.bias_vectors.append(bias_vector)  # 添加到偏置向量列表

    def forward(self, input_bow, theta_list, beta_list):  # 定义前向传播方法
        topk_bias_list = list()  # 初始化top-k偏置列表
        all_recon_loss = 0.  # 初始化总重构损失

        for layer_id in range(self.num_layers):  # 遍历每一层
            topk_bias = utils.get_topk_tensor(beta_list[layer_id], topk=self.bias_topk).sum(0)  # 获取该层的top-k偏置
            topk_bias = topk_bias.detach()  # 分离梯度
            topk_bias_list.append(topk_bias)  # 添加到偏置列表

        for layer_id in range(self.num_layers):  # 再次遍历每一层
            topk_bias = 0.  # 初始化top-k偏置
            # previous layer  # 前一层
            if layer_id > 0:  # 如果不是第一层
                topk_bias += topk_bias_list[layer_id - 1]  # 加上前一层的偏置

            # next layer  # 后一层
            if layer_id < self.num_layers - 1:  # 如果不是最后一层
                topk_bias += topk_bias_list[layer_id + 1]  # 加上后一层的偏置

            topk_mask = (topk_bias > 0).float()  # 创建top-k掩码
            bias = self.bias_p * topk_bias * topk_mask + self.bias_vectors[layer_id] * (1 - topk_mask)  # 计算偏置

            recon = self.decoder_bn_list[layer_id](torch.matmul(theta_list[layer_id], beta_list[layer_id]))  # 计算重构结果并批归一化
            recon = recon + bias  # 加上偏置
            recon = F.softmax(recon, dim=-1)  # 应用softmax归一化
            recon_loss = -(input_bow * (recon + 1e-12).log()).sum(axis=1)  # 计算重构损失
            recon_loss = recon_loss.mean()  # 计算平均重构损失
            all_recon_loss += recon_loss  # 累加重构损失

        all_recon_loss /= self.num_layers  # 计算平均重构损失

        return all_recon_loss  # 返回总重构损失
