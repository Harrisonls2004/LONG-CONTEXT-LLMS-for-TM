import torch  # 导入PyTorch库


def sinkhorn_loss(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2):  # 定义Sinkhorn损失函数
    """
    计算Sinkhorn散度损失，用于最优传输问题

    参数:
        M: 成本矩阵
        a: 源分布
        b: 目标分布
        lambda_sh: 正则化参数
        numItermax: 最大迭代次数，默认5000
        stopThr: 停止阈值，默认0.005

    返回:
        sinkhorn_divergences: Sinkhorn散度
    """
    device = a.device  # 获取张量所在设备

    u = (torch.ones_like(a) / a.size()[0]).to(device)  # 初始化对偶变量u，归一化为均匀分布
    # TODO v is zeros in the tensorflow code.  # 待办：在TensorFlow代码中v初始化为零
    # v = (torch.ones_like(b)).to(device)  # 注释掉的v初始化

    K = torch.exp(-M * lambda_sh)  # 计算核矩阵K，使用指数函数和正则化参数
    err = 1  # 初始化误差为1
    cpt = 0  # 初始化计数器为0
    while err > stopThr and cpt < numItermax:  # 当误差大于阈值且未达到最大迭代次数时继续
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))  # 更新对偶变量u
        cpt += 1  # 计数器加1
        if cpt % 20 == 1:  # 每20次迭代检查一次收敛性
            v = torch.div(b, torch.matmul(K.t(), u))  # 计算对偶变量v
            u = torch.div(a, torch.matmul(K, v))  # 重新计算对偶变量u
            bb = torch.mul(v, torch.matmul(K.t(), u))  # 计算重构的目标分布
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))  # 计算无穷范数误差

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)  # 计算最终的Sinkhorn散度

    return sinkhorn_divergences  # 返回Sinkhorn散度
