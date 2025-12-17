

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class ETM(nn.Module):  # 定义嵌入主题模型类
    '''
        Topic Modeling in Embedding Spaces. TACL 2020  # 嵌入空间中的主题建模，TACL 2020

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.  # 作者信息
    '''
    def __init__(self, vocab_size, embed_size=200, num_topics=50, en_units=800, dropout=0., pretrained_WE=None, train_WE=False):  # 初始化方法
        super().__init__()  # 调用父类初始化

        if pretrained_WE is not None:  # 如果有预训练词嵌入
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())  # 使用预训练词嵌入
        else:  # 否则
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))  # 随机初始化词嵌入

        self.word_embeddings.requires_grad = train_WE  # 设置词嵌入是否参与训练

        self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))  # 初始化主题嵌入

        self.encoder1 = nn.Sequential(  # 定义编码器
            nn.Linear(vocab_size, en_units),  # 第一层全连接
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(en_units, en_units),  # 第二层全连接
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )

        self.fc21 = nn.Linear(en_units, num_topics)  # 均值输出层
        self.fc22 = nn.Linear(en_units, num_topics)  # 方差输出层

    def reparameterize(self, mu, logvar):  # 定义重参数化方法
        if self.training:  # 如果处于训练模式
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成随机噪声
            return mu + (eps * std)  # 返回重参数化结果
        else:  # 如果处于测试模式
            return mu  # 直接返回均值

    def encode(self, x):  # 定义编码方法
        e1 = self.encoder1(x)  # 编码输入
        return self.fc21(e1), self.fc22(e1)  # 返回均值和方差

    def get_theta(self, x):  # 定义获取主题分布方法
        # Warn: normalize the input if use Relu.  # 警告：如果使用ReLU，需要归一化输入
        # https://github.com/adjidieng/ETM/issues/3  # GitHub问题链接
        norm_x = x / x.sum(1, keepdim=True)  # 归一化输入
        mu, logvar = self.encode(norm_x)  # 编码获取均值和方差
        z = self.reparameterize(mu, logvar)  # 重参数化
        theta = F.softmax(z, dim=-1)  # 应用softmax归一化
        if self.training:  # 如果处于训练模式
            return theta, mu, logvar  # 返回主题分布、均值、方差
        else:  # 如果处于测试模式
            return theta  # 只返回主题分布

    def get_beta(self):  # 定义获取主题-词分布方法
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1)  # 计算主题-词分布
        return beta  # 返回主题-词分布

    def forward(self, x, avg_loss=True):  # 定义前向传播方法
        theta, mu, logvar = self.get_theta(x)  # 获取主题分布和相关参数
        beta = self.get_beta()  # 获取主题-词分布
        recon_x = torch.matmul(theta, beta)  # 重构输入

        loss = self.loss_function(x, recon_x, mu, logvar, avg_loss)  # 计算损失
        return {'loss': loss}  # 返回损失字典

    def loss_function(self, x, recon_x, mu, logvar, avg_loss=True):  # 定义损失函数
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)  # 计算重构损失
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)  # 计算KL散度
        loss = (recon_loss + KLD)  # 计算总损失

        if avg_loss:  # 如果需要平均损失
            loss = loss.mean()  # 计算平均损失

        return loss  # 返回损失
