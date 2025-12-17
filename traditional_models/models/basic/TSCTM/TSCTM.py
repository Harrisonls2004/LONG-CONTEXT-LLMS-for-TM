
# 空行，用于代码格式化
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
from .TopicDistQuant import TopicDistQuant  # 导入主题分布量化模块
from .TSC import TSC  # 导入主题-语义对比学习模块


class TSCTM(nn.Module):  # 定义TSCTM（主题-语义对比主题模型）类
    '''
        Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning. EMNLP 2022
        通过主题-语义对比学习缓解短文本主题建模中的数据稀疏性问题，EMNLP 2022

        Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong.  # 作者信息

        Note: This implementation does not include TSCTM with augmentations. For augmentations, see https://github.com/BobXWu/TSCTM.
        注意：此实现不包含带数据增强的TSCTM。有关数据增强，请参见 https://github.com/BobXWu/TSCTM
    '''

    def __init__(self, vocab_size, num_topics=50, en_units=200, temperature=0.5, weight_contrast=1.0):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.fc11 = nn.Linear(vocab_size, en_units)  # 第一层全连接层：词汇表大小到编码单元数
        self.fc12 = nn.Linear(en_units, en_units)  # 第二层全连接层：编码单元到编码单元
        self.fc21 = nn.Linear(en_units, num_topics)  # 第三层全连接层：编码单元到主题数

        self.mean_bn = nn.BatchNorm1d(num_topics)  # 主题分布的批归一化层
        self.mean_bn.weight.requires_grad = False  # 批归一化权重不参与训练
        self.decoder_bn = nn.BatchNorm1d(vocab_size)  # 解码器的批归一化层
        self.decoder_bn.weight.requires_grad = False  # 批归一化权重不参与训练

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)  # 解码器全连接层：主题数到词汇表大小，无偏置

        for m in self.modules():  # 遍历所有模块
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # 如果是卷积层或全连接层
                nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀分布初始化权重
                if m.bias is not None:  # 如果有偏置项
                    nn.init.zeros_(m.bias)  # 将偏置初始化为零

        self.topic_dist_quant = TopicDistQuant(num_topics, num_topics)  # 创建主题分布量化模块
        self.contrast_loss = TSC(temperature, weight_contrast)  # 创建主题-语义对比损失模块

    def get_beta(self):  # 定义获取主题-词分布方法
        return self.fcd1.weight.T  # 返回解码器权重的转置作为主题-词分布

    def encode(self, inputs):  # 定义编码方法
        e1 = F.softplus(self.fc11(inputs))  # 第一层全连接后应用Softplus激活函数
        e1 = F.softplus(self.fc12(e1))  # 第二层全连接后应用Softplus激活函数
        return self.mean_bn(self.fc21(e1))  # 第三层全连接后应用批归一化并返回

    def decode(self, theta):  # 定义解码方法
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)  # 解码器全连接、批归一化后应用Softmax
        return d1  # 返回重构的词分布

    def get_theta(self, inputs):  # 定义获取主题分布方法
        theta = self.encode(inputs)  # 编码输入获取主题表示
        softmax_theta = F.softmax(theta, dim=1)  # 应用Softmax获取主题分布
        return softmax_theta  # 返回主题分布

    def forward(self, inputs):  # 定义前向传播方法
        theta = self.encode(inputs)  # 编码输入获取主题表示
        softmax_theta = F.softmax(theta, dim=1)  # 应用Softmax获取主题分布

        quant_rst = self.topic_dist_quant(softmax_theta)  # 对主题分布进行量化

        recon = self.decode(quant_rst['quantized'])  # 使用量化后的主题分布进行解码重构
        loss = self.loss_function(recon, inputs) + quant_rst['loss']  # 计算重构损失和量化损失

        features = torch.cat([F.normalize(theta, dim=1).unsqueeze(1)], dim=1)  # 归一化主题表示并增加维度作为特征
        contrastive_loss = self.contrast_loss(features, quant_idx=quant_rst['encoding_indices'])  # 计算对比损失
        loss += contrastive_loss  # 将对比损失加入总损失

        return {'loss': loss, 'contrastive_loss': contrastive_loss}  # 返回损失字典

    def loss_function(self, recon_x, x):  # 定义损失函数
        loss = -(x * (recon_x).log()).sum(axis=1)  # 计算负对数似然损失（重构损失）
        loss = loss.mean()  # 计算批次平均损失
        return loss  # 返回损失值
