import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class MLPEncoder(nn.Module):  # 定义多层感知机编码器类
    def __init__(self, vocab_size, num_topic, hidden_dim, dropout):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.fc11 = nn.Linear(vocab_size, hidden_dim)  # 第一层全连接层
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)  # 第二层全连接层
        self.fc21 = nn.Linear(hidden_dim, num_topic)  # 均值输出层
        self.fc22 = nn.Linear(hidden_dim, num_topic)  # 方差输出层

        self.fc1_drop = nn.Dropout(dropout)  # 全连接层dropout
        self.z_drop = nn.Dropout(dropout)  # 潜在变量dropout

        self.mean_bn = nn.BatchNorm1d(num_topic, affine=True)  # 均值批归一化
        self.mean_bn.weight.requires_grad = False  # 均值批归一化权重不参与训练
        self.logvar_bn = nn.BatchNorm1d(num_topic, affine=True)  # 方差批归一化
        self.logvar_bn.weight.requires_grad = False  # 方差批归一化权重不参与训练

    def reparameterize(self, mu, logvar):  # 定义重参数化方法
        if self.training:  # 如果处于训练模式
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成随机噪声
            return mu + (eps * std)  # 返回重参数化结果
        else:  # 如果处于测试模式
            return mu  # 直接返回均值

    def forward(self, x):  # 定义前向传播方法
        e1 = F.softplus(self.fc11(x))  # 第一层全连接并激活
        e1 = F.softplus(self.fc12(e1))  # 第二层全连接并激活
        e1 = self.fc1_drop(e1)  # 应用dropout
        mu = self.mean_bn(self.fc21(e1))  # 计算均值并批归一化
        logvar = self.logvar_bn(self.fc22(e1))  # 计算方差并批归一化
        theta = self.reparameterize(mu, logvar)  # 重参数化
        theta = F.softmax(theta, dim=1)  # 应用softmax归一化
        theta = self.z_drop(theta)  # 应用dropout
        return theta, mu, logvar  # 返回主题分布、均值、方差
