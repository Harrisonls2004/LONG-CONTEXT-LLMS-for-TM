

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
import numpy as np  # 导入numpy库


class ProdLDA(nn.Module):  # 定义ProdLDA模型类
    '''
        Autoencoding Variational Inference For Topic Models. ICLR 2017  # 主题模型的自动编码变分推理，ICLR 2017

        Akash Srivastava, Charles Sutton.  # 作者信息
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.num_topics = num_topics  # 设置主题数量

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)  # 初始化Dirichlet先验参数
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))  # 初始化均值参数
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))  # 初始化方差参数

        self.mu2.requires_grad = False  # 均值参数不参与训练
        self.var2.requires_grad = False  # 方差参数不参与训练

        self.fc11 = nn.Linear(vocab_size, en_units)  # 第一层全连接层
        self.fc12 = nn.Linear(en_units, en_units)  # 第二层全连接层
        self.fc21 = nn.Linear(en_units, num_topics)  # 均值输出层
        self.fc22 = nn.Linear(en_units, num_topics)  # 方差输出层

        # align with the default parameters of tf.contrib.layers.batch_norm  # 与TensorFlow批归一化默认参数对齐
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/batch_norm  # TensorFlow文档链接
        # center=True (add bias(beta)), scale=False (weight(gamma) is not used)  # center=True（添加偏置），scale=False（不使用权重）
        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)  # 均值批归一化
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))  # 初始化权重为1
        self.mean_bn.weight.requires_grad = False  # 权重不参与训练

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)  # 方差批归一化
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))  # 初始化权重为1
        self.logvar_bn.weight.requires_grad = False  # 权重不参与训练

        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)  # 解码器批归一化
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))  # 初始化权重为1
        self.decoder_bn.weight.requires_grad = False  # 权重不参与训练

        self.fc1_drop = nn.Dropout(dropout)  # 全连接层dropout
        self.theta_drop = nn.Dropout(dropout)  # 主题分布dropout

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)  # 解码器全连接层
        nn.init.xavier_uniform_(self.fcd1.weight)  # Xavier初始化权重

    def get_beta(self):  # 定义获取主题-词分布方法
        return self.fcd1.weight.T  # 返回主题-词分布矩阵的转置

    def get_theta(self, x):  # 定义获取主题分布方法
        mu, logvar = self.encode(x)  # 编码获取均值和方差
        z = self.reparameterize(mu, logvar)  # 重参数化
        theta = F.softmax(z, dim=1)  # 应用softmax归一化
        theta = self.theta_drop(theta)  # 应用dropout
        if self.training:  # 如果处于训练模式
            return theta, mu, logvar  # 返回主题分布、均值、方差
        else:  # 如果处于测试模式
            return theta  # 只返回主题分布

    def reparameterize(self, mu, logvar):  # 定义重参数化方法
        if self.training:  # 如果处于训练模式
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成随机噪声
            return mu + (eps * std)  # 返回重参数化结果
        else:  # 如果处于测试模式
            return mu  # 直接返回均值

    def encode(self, x):  # 定义编码方法
        e1 = F.softplus(self.fc11(x))  # 第一层全连接并激活
        e1 = F.softplus(self.fc12(e1))  # 第二层全连接并激活
        e1 = self.fc1_drop(e1)  # 应用dropout
        return self.mean_bn(self.fc21(e1)), self.logvar_bn(self.fc22(e1))  # 返回批归一化后的均值和方差

    def decode(self, theta):  # 定义解码方法
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)  # 解码并应用softmax
        return d1  # 返回重构结果

    def forward(self, x):  # 定义前向传播方法
        theta, mu, logvar = self.get_theta(x)  # 获取主题分布和相关参数
        recon_x = self.decode(theta)  # 解码重构输入
        loss = self.loss_function(x, recon_x, mu, logvar)  # 计算损失
        return {'loss': loss}  # 返回损失字典

    def loss_function(self, x, recon_x, mu, logvar):  # 定义损失函数
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)  # 计算重构损失
        var = logvar.exp()  # 计算方差
        var_division = var / self.var2  # 方差比值
        diff = mu - self.mu2  # 均值差
        diff_term = diff * diff / self.var2  # 均值差项
        logvar_division = self.var2.log() - logvar  # 对数方差差
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)  # 计算KL散度
        loss = (recon_loss + KLD).mean()  # 计算总损失
        return loss  # 返回损失
