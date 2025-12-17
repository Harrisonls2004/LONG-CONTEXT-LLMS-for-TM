import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class ECR(nn.Module):  # 定义嵌入聚类正则化类
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023
        基于嵌入聚类正则化的有效神经主题建模，ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.  # 作者信息
    '''
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.sinkhorn_alpha = sinkhorn_alpha  # 设置Sinkhorn算法的alpha参数
        self.OT_max_iter = OT_max_iter  # 设置最优传输的最大迭代次数
        self.weight_loss_ECR = weight_loss_ECR  # 设置ECR损失的权重
        self.stopThr = stopThr  # 设置停止阈值
        self.epsilon = 1e-16  # 设置数值稳定性的小常数

    def forward(self, M):  # 定义前向传播方法
        # M: KxV  # M矩阵：主题数x词汇表大小
        # a: Kx1  # a向量：主题数x1
        # b: Vx1  # b向量：词汇表大小x1
        device = M.device  # 获取张量所在设备

        # Sinkhorn's algorithm  # Sinkhorn算法
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)  # 初始化均匀分布a
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)  # 初始化均匀分布b

        u = (torch.ones_like(a) / a.size()[0]).to(device)  # 初始化对偶变量u，Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)  # 计算核矩阵K
        err = 1  # 初始化误差为1
        cpt = 0  # 初始化计数器为0
        while err > self.stopThr and cpt < self.OT_max_iter:  # 当误差大于阈值且未达到最大迭代次数时继续
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)  # 更新对偶变量v
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)  # 更新对偶变量u
            cpt += 1  # 计数器加1
            if cpt % 50 == 1:  # 每50次迭代检查一次收敛性
                bb = torch.mul(v, torch.matmul(K.t(), u))  # 计算重构的目标分布
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))  # 计算无穷范数误差

        transp = u * (K * v.T)  # 计算传输矩阵

        loss_ECR = torch.sum(transp * M)  # 计算ECR损失
        loss_ECR *= self.weight_loss_ECR  # 乘以权重

        return loss_ECR  # 返回ECR损失
