import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
import numpy as np  # 导入numpy库


class NMTM(nn.Module):  # 定义神经多语言主题模型类
    '''
        Learning Multilingual Topics with Neural Variational Inference. NLPCC 2020.
        基于神经变分推理的多语言主题学习，NLPCC 2020

        Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao.  # 作者信息
    '''
    def __init__(self, Map_en2cn, Map_cn2en, vocab_size_en, vocab_size_cn, num_topics=50, en_units=200, dropout=0., lam=0.8):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.num_topics = num_topics  # 设置主题数量
        self.lam = lam  # 设置语言混合参数

        # V_en x V_cn  # 英文词汇表大小 x 中文词汇表大小
        self.Map_en2cn = nn.Parameter(torch.as_tensor(Map_en2cn).float(), requires_grad=False)  # 英文到中文的映射矩阵

        # V_cn x V_en  # 中文词汇表大小 x 英文词汇表大小
        self.Map_cn2en = nn.Parameter(torch.as_tensor(Map_cn2en).float(), requires_grad=False)  # 中文到英文的映射矩阵

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)  # 初始化Dirichlet先验参数
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)  # 初始化均值参数
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)  # 初始化方差参数

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, eps=0.001, momentum=0.001, affine=True)  # 英文解码器批归一化
        self.decoder_bn_en.weight.requires_grad = False  # 权重不参与训练

        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, eps=0.001, momentum=0.001, affine=True)  # 中文解码器批归一化
        self.decoder_bn_cn.weight.requires_grad = False  # 权重不参与训练

        self.fc11_en = nn.Linear(vocab_size_en, en_units)  # 英文第一层全连接层
        self.fc11_cn = nn.Linear(vocab_size_cn, en_units)  # 中文第一层全连接层
        self.fc12 = nn.Linear(en_units, en_units)  # 第二层全连接层
        self.fc21 = nn.Linear(en_units, num_topics)  # 均值输出层
        self.fc22 = nn.Linear(en_units, num_topics)  # 方差输出层

        self.fc1_drop = nn.Dropout(dropout)  # 全连接层dropout
        self.z_drop = nn.Dropout(dropout)  # 潜在变量dropout

        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)  # 均值批归一化
        self.mean_bn.weight.requires_grad = False  # 权重不参与训练

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)  # 方差批归一化
        self.logvar_bn.weight.requires_grad = False  # 权重不参与训练

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))  # 英文主题-词分布参数
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_cn))))  # 中文主题-词分布参数

    def reparameterize(self, mu, logvar):  # 定义重参数化方法
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 生成随机噪声
        return eps.mul(std).add_(mu)  # 返回重参数化结果

    def encode(self, x, lang):  # 定义编码方法
        e1 = F.softplus(getattr(self, f'fc11_{lang}')(x))  # 根据语言选择对应的第一层全连接层并激活

        e1 = F.softplus(self.fc12(e1))  # 第二层全连接并激活
        e1 = self.fc1_drop(e1)  # 应用dropout
        mu = self.mean_bn(self.fc21(e1))  # 计算均值并批归一化
        logvar = self.logvar_bn(self.fc22(e1))  # 计算方差并批归一化
        theta = self.reparameterize(mu, logvar)  # 重参数化
        theta = F.softmax(theta, dim=1)  # 应用softmax归一化
        theta = self.z_drop(theta)  # 应用dropout
        return theta, mu, logvar  # 返回主题分布、均值、方差

    def get_theta(self, x, lang):  # 定义获取主题分布方法
        theta, mu, logvar = self.encode(x, lang)  # 编码获取主题分布和相关参数

        if self.training:  # 如果处于训练模式
            return theta, mu, logvar  # 返回主题分布、均值、方差
        else:  # 如果处于测试模式
            return mu  # 只返回均值

    def get_beta(self):  # 定义获取主题-词分布方法
        beta_en = self.lam * torch.matmul(self.phi_cn, self.Map_cn2en) + (1 - self.lam) * self.phi_en  # 计算英文主题-词分布
        beta_cn = self.lam * torch.matmul(self.phi_en, self.Map_en2cn) + (1 - self.lam) * self.phi_cn  # 计算中文主题-词分布
        return beta_en, beta_cn  # 返回双语主题-词分布

    def decode(self, theta, lang):  # 定义解码方法
        d1 = F.softmax(getattr(self, f'decoder_bn_{lang}')(torch.matmul(theta, getattr(self, f'beta_{lang}'))), dim=1)  # 根据语言解码并应用softmax
        return d1  # 返回重构结果

    def forward(self, x_en, x_cn):  # 定义前向传播方法
        self.beta_en, self.beta_cn = self.get_beta()  # 获取双语主题-词分布

        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')  # 获取英文主题分布和相关参数
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')  # 获取中文主题分布和相关参数

        x_recon_en = self.decode(theta_en, lang='en')  # 解码重构英文输入
        x_recon_cn = self.decode(theta_cn, lang='cn')  # 解码重构中文输入

        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)  # 计算英文损失
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)  # 计算中文损失

        loss = loss_en + loss_cn  # 计算总损失

        rst_dict = {  # 构建返回结果字典
            'loss': loss  # 损失
        }

        return rst_dict  # 返回结果字典

    def loss_function(self, recon_x, x, mu, logvar):  # 定义损失函数
        var = logvar.exp()  # 计算方差
        var_division = var / self.var2  # 方差比值
        diff = mu - self.mu2  # 均值差
        diff_term = diff * diff / self.var2  # 均值差项
        logvar_division = self.var2.log() - logvar  # 对数方差差
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)  # 计算KL散度

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)  # 计算重构损失

        LOSS = (RECON + KLD).mean()  # 计算总损失
        return LOSS  # 返回损失
