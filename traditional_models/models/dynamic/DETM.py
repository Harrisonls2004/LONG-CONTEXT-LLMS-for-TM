
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class DETM(nn.Module):  # 定义动态嵌入主题模型类
    """
        The Dynamic Embedded Topic Model. 2019
        动态嵌入主题模型，2019

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei  # 作者信息
    """
    def __init__(self, vocab_size, num_times, train_size, train_time_wordfreq, num_topics=50, train_WE=True, pretrained_WE=None, en_units=800, eta_hidden_size=200, rho_size=300, enc_drop=0.0, eta_nlayers=3, eta_dropout=0.0, delta=0.005, theta_act='relu', device='cpu'):  # 初始化方法
        super().__init__()  # 调用父类初始化

        ## define hyperparameters  # 定义超参数
        self.num_topics = num_topics  # 设置主题数量
        self.num_times = num_times  # 设置时间步数
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.eta_hidden_size = eta_hidden_size  # 设置eta隐藏层大小
        self.rho_size = rho_size  # 设置rho嵌入大小
        self.enc_drop = enc_drop  # 设置编码器dropout率
        self.eta_nlayers = eta_nlayers  # 设置eta网络层数
        self.t_drop = nn.Dropout(enc_drop)  # 创建dropout层
        self.eta_dropout = eta_dropout  # 设置eta dropout率
        self.delta = delta  # 设置时间演化参数
        self.train_WE = train_WE  # 设置是否训练词嵌入
        self.train_size = train_size  # 设置训练集大小
        self.rnn_inp = train_time_wordfreq  # 设置RNN输入（时间词频）
        self.device = device  # 设置设备

        self.theta_act = self.get_activation(theta_act)  # 获取theta激活函数

        ## define the word embedding matrix \rho  # 定义词嵌入矩阵rho
        if self.train_WE:  # 如果训练词嵌入
            self.rho = nn.Linear(self.rho_size, self.vocab_size, bias=False)  # 创建可训练的词嵌入层
        else:  # 否则
            rho = nn.Embedding(pretrained_WE.size())  # 创建嵌入层
            rho.weight.data = torch.from_numpy(pretrained_WE)  # 加载预训练词嵌入
            self.rho = rho.weight.data.clone().float().to(self.device)  # 复制并转移到设备

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L  # 定义主题嵌入随时间变化的变分参数alpha，维度为K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))  # alpha的均值参数
        self.logsigma_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))  # alpha的对数方差参数

        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D  # 通过摊销推理定义theta的变分分布，维度为K x D
        self.q_theta = nn.Sequential(  # theta的编码器网络
            nn.Linear(self.vocab_size + self.num_topics, en_units),  # 第一层全连接
            self.theta_act,  # 激活函数
            nn.Linear(en_units, en_units),  # 第二层全连接
            self.theta_act,  # 激活函数
        )
        self.mu_q_theta = nn.Linear(en_units, self.num_topics, bias=True)  # theta均值输出层
        self.logsigma_q_theta = nn.Linear(en_units, self.num_topics, bias=True)  # theta对数方差输出层

        ## define variational distribution for \eta via amortizartion... eta is K x T  # 通过摊销推理定义eta的变分分布，维度为K x T
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)  # eta映射层
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)  # eta的LSTM网络
        self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)  # eta均值输出层
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)  # eta对数方差输出层

        self.decoder_bn = nn.BatchNorm1d(vocab_size)  # 解码器批归一化
        self.decoder_bn.weight.requires_grad = False  # 权重不参与训练

    def get_activation(self, act):  # 定义获取激活函数方法
        activations = {  # 激活函数字典
            'tanh': nn.Tanh(),  # 双曲正切
            'relu': nn.ReLU(),  # ReLU
            'softplus': nn.Softplus(),  # Softplus
            'rrelu': nn.RReLU(),  # 随机ReLU
            'leakyrelu': nn.LeakyReLU(),  # 泄漏ReLU
            'elu': nn.ELU(),  # ELU
            'selu': nn.SELU(),  # SELU
            'glu': nn.GLU(),  # GLU
        }

        if act in activations:  # 如果激活函数在字典中
            act = activations[act]  # 获取对应的激活函数
        else:  # 否则
            print('Defaulting to tanh activations...')  # 打印默认信息
            act = nn.Tanh()  # 使用默认的Tanh激活函数
        return act  # 返回激活函数

    def reparameterize(self, mu, logvar):  # 定义重参数化方法
        """Returns a sample from a Gaussian distribution via reparameterization.
        通过重参数化从高斯分布中采样
        """
        if self.training:  # 如果处于训练模式
            std = torch.exp(0.5 * logvar)  # 计算标准差
            eps = torch.randn_like(std)  # 生成随机噪声
            return eps.mul_(std).add_(mu)  # 返回重参数化结果
        else:  # 如果处于测试模式
            return mu  # 直接返回均值

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):  # 定义计算KL散度方法
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        计算两个高斯分布之间的KL散度
        """
        if p_mu is not None and p_logsigma is not None:  # 如果提供了先验分布参数
            sigma_q_sq = torch.exp(q_logsigma)  # 计算后验方差
            sigma_p_sq = torch.exp(p_logsigma)  # 计算先验方差
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )  # 计算KL散度第一项
            kl = kl - 1 + p_logsigma - q_logsigma  # 计算KL散度第二项
            kl = 0.5 * torch.sum(kl, dim=-1)  # 求和并乘以0.5
        else:  # 如果没有提供先验分布参数（标准高斯先验）
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)  # 计算标准KL散度
        return kl  # 返回KL散度

    def get_alpha(self):  # 定义获取alpha（主题嵌入）方法，使用平均场变分推理
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)  # 初始化alpha张量
        kl_alpha = []  # 初始化KL散度列表

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])  # 采样第一个时间步的alpha

        # TODO: why logsigma_p_0 is zero?  # 待办：为什么初始对数方差为零？
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)  # 初始先验均值
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)  # 初始先验对数方差
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)  # 计算第一个时间步的KL散度
        kl_alpha.append(kl_0)  # 添加到KL散度列表
        for t in range(1, self.num_times):  # 遍历其余时间步
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])  # 采样当前时间步的alpha

            p_mu_t = alphas[t - 1]  # 先验均值为前一时间步的alpha
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))  # 先验对数方差
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)  # 计算当前时间步的KL散度
            kl_alpha.append(kl_t)  # 添加到KL散度列表
        kl_alpha = torch.stack(kl_alpha).sum()  # 堆叠并求和KL散度
        return alphas, kl_alpha.sum()  # 返回alpha和总KL散度

    def get_eta(self, rnn_inp):  # 定义获取eta方法，使用结构化摊销推理
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)  # 映射RNN输入并增加维度
        hidden = self.init_hidden()  # 初始化隐藏状态
        output, _ = self.q_eta(inp, hidden)  # 通过LSTM获取输出
        output = output.squeeze()  # 压缩维度

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)  # 初始化eta张量
        kl_eta = []  # 初始化KL散度列表

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)  # 第一个时间步的输入
        mu_0 = self.mu_q_eta(inp_0)  # 计算第一个时间步的均值
        logsigma_0 = self.logsigma_q_eta(inp_0)  # 计算第一个时间步的对数方差
        etas[0] = self.reparameterize(mu_0, logsigma_0)  # 采样第一个时间步的eta

        p_mu_0 = torch.zeros(self.num_topics,).to(self.device)  # 初始先验均值
        logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)  # 初始先验对数方差
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)  # 计算第一个时间步的KL散度
        kl_eta.append(kl_0)  # 添加到KL散度列表

        for t in range(1, self.num_times):  # 遍历其余时间步
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)  # 当前时间步的输入（包含前一时间步的eta）
            mu_t = self.mu_q_eta(inp_t)  # 计算当前时间步的均值
            logsigma_t = self.logsigma_q_eta(inp_t)  # 计算当前时间步的对数方差
            etas[t] = self.reparameterize(mu_t, logsigma_t)  # 采样当前时间步的eta

            p_mu_t = etas[t-1]  # 先验均值为前一时间步的eta
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))  # 先验对数方差
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)  # 计算当前时间步的KL散度
            kl_eta.append(kl_t)  # 添加到KL散度列表
        kl_eta = torch.stack(kl_eta).sum()  # 堆叠并求和KL散度

        return etas, kl_eta  # 返回eta和总KL散度

    def get_theta(self, bows, times, eta=None):  # 定义获取theta方法，使用摊销推理
        """Returns the topic proportions.
        返回主题比例
        """

        normalized_bows = bows / bows.sum(1, keepdims=True)  # 归一化词袋向量

        if eta is None and self.training is False:  # 如果eta为空且处于测试模式
            eta, kl_eta = self.get_eta(self.rnn_inp)  # 获取eta

        eta_td = eta[times]  # 根据时间索引获取对应的eta
        inp = torch.cat([normalized_bows, eta_td], dim=1)  # 连接归一化词袋和eta
        q_theta = self.q_theta(inp)  # 通过编码器获取theta的隐藏表示
        if self.enc_drop > 0:  # 如果dropout率大于0
            q_theta = self.t_drop(q_theta)  # 应用dropout
        mu_theta = self.mu_q_theta(q_theta)  # 计算theta的均值
        logsigma_theta = self.logsigma_q_theta(q_theta)  # 计算theta的对数方差
        z = self.reparameterize(mu_theta, logsigma_theta)  # 重参数化采样
        theta = F.softmax(z, dim=-1)  # 应用softmax获取主题分布
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))  # 计算KL散度

        if self.training:  # 如果处于训练模式
            return theta, kl_theta  # 返回主题分布和KL散度
        else:  # 如果处于测试模式
            return theta  # 只返回主题分布

    @property  # 属性装饰器
    def word_embeddings(self):  # 定义词嵌入属性
        return self.rho.weight  # 返回词嵌入权重

    @property  # 属性装饰器
    def topic_embeddings(self):  # 定义主题嵌入属性
        alpha, _ = self.get_alpha()  # 获取alpha（主题嵌入）
        return alpha  # 返回主题嵌入

    def get_beta(self, alpha=None):  # 定义获取beta（主题-词分布）方法
        """Returns the topic matrix \beta of shape T x K x V
        返回形状为T x K x V的主题矩阵beta
        """

        if alpha is None and self.training is False:  # 如果alpha为空且处于测试模式
            alpha, kl_alpha = self.get_alpha()  # 获取alpha

        if self.train_WE:  # 如果训练词嵌入
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))  # 通过线性层计算logits
        else:  # 否则
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)  # 重塑alpha
            logit = torch.mm(tmp, self.rho.permute(1, 0))  # 矩阵乘法计算logits
        logit = logit.view(alpha.size(0), alpha.size(1), -1)  # 重塑logits

        beta = F.softmax(logit, dim=-1)  # 应用softmax获取主题-词分布

        return beta  # 返回beta

    def get_NLL(self, theta, beta, bows):  # 定义计算负对数似然方法
        theta = theta.unsqueeze(1)  # 增加theta的维度
        loglik = torch.bmm(theta, beta).squeeze(1)  # 批量矩阵乘法计算似然
        loglik = torch.log(loglik + 1e-12)  # 计算对数似然
        nll = -loglik * bows  # 计算负对数似然
        nll = nll.sum(-1)  # 求和
        return nll  # 返回负对数似然

    def forward(self, bows, times):  # 定义前向传播方法
        bsz = bows.size(0)  # 获取批次大小
        coeff = self.train_size / bsz  # 计算系数
        eta, kl_eta = self.get_eta(self.rnn_inp)  # 获取eta和对应的KL散度
        theta, kl_theta = self.get_theta(bows, times, eta)  # 获取theta和对应的KL散度
        kl_theta = kl_theta.sum() * coeff  # 计算theta的总KL散度

        alpha, kl_alpha = self.get_alpha()  # 获取alpha和对应的KL散度
        beta = self.get_beta(alpha)  # 获取beta

        beta = beta[times]  # 根据时间索引获取对应的beta
        # beta = beta[times.type('torch.LongTensor')]  # 注释掉的类型转换
        nll = self.get_NLL(theta, beta, bows)  # 计算负对数似然
        nll = nll.sum() * coeff  # 计算总负对数似然

        loss = nll + kl_eta + kl_theta  # 计算损失

        rst_dict = {  # 构建返回结果字典
            'loss': loss,  # 损失
            'nll': nll,  # 负对数似然
            'kl_eta': kl_eta,  # eta的KL散度
            'kl_theta': kl_theta  # theta的KL散度
        }

        loss += kl_alpha  # 加上alpha的KL散度
        rst_dict['kl_alpha'] = kl_alpha  # 添加alpha的KL散度到结果字典

        return rst_dict  # 返回结果字典

    def init_hidden(self):  # 定义初始化隐藏状态方法
        """Initializes the first hidden state of the RNN used as inference network for \\eta.
        初始化用于eta推理网络的RNN的第一个隐藏状态
        """
        weight = next(self.parameters())  # 获取模型参数
        nlayers = self.eta_nlayers  # 获取层数
        nhid = self.eta_hidden_size  # 获取隐藏层大小
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))  # 返回初始化的隐藏状态和细胞状态
