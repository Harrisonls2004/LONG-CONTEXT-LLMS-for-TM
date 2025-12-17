import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class ETC(nn.Module):  # 定义嵌入时间对比学习类
    def __init__(self, num_times, temperature, weight_neg, weight_pos):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.num_times = num_times  # 设置时间步数
        self.weight_neg = weight_neg  # 设置负样本权重
        self.weight_pos = weight_pos  # 设置正样本权重
        self.temperature = temperature  # 设置温度参数

    def forward(self, topic_embeddings):  # 定义前向传播方法
        loss = 0.  # 初始化总损失
        loss_neg = 0.  # 初始化负样本损失
        loss_pos = 0.  # 初始化正样本损失

        for t in range(self.num_times):  # 遍历每个时间步
            loss_neg += self.compute_loss(topic_embeddings[t], topic_embeddings[t], self.temperature, self_contrast=True)  # 计算自对比负样本损失

        for t in range(1, self.num_times):  # 从第二个时间步开始遍历
            loss_pos += self.compute_loss(topic_embeddings[t], topic_embeddings[t - 1].detach(), self.temperature, self_contrast=False, only_pos=True)  # 计算相邻时间步的正样本损失

        loss_neg *= (self.weight_neg / self.num_times)  # 归一化负样本损失
        loss_pos *= (self.weight_pos / (self.num_times - 1))  # 归一化正样本损失
        loss = loss_neg + loss_pos  # 计算总损失

        return loss  # 返回总损失

    def compute_loss(self, anchor_feature, contrast_feature, temperature, self_contrast=False, only_pos=False, all_neg=False):  # 定义计算损失方法
        # KxK  # 主题数x主题数
        anchor_dot_contrast = torch.div(  # 计算锚点与对比特征的点积
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),  # 归一化后的矩阵乘法
            temperature  # 除以温度参数
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 找到最大值
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值以提高数值稳定性

        pos_mask = torch.eye(anchor_dot_contrast.shape[0]).to(anchor_dot_contrast.device)  # 创建正样本掩码（单位矩阵）

        if self_contrast is False:  # 如果不是自对比
            if only_pos is False:  # 如果不是只有正样本
                if all_neg is True:  # 如果是全负样本
                    exp_logits = torch.exp(logits)  # 计算指数logits
                    sum_exp_logits = exp_logits.sum(1)  # 计算指数logits的和
                    log_prob = -torch.log(sum_exp_logits + 1e-12)  # 计算对数概率

                    mean_log_prob = -log_prob.sum() / (logits.shape[0] * logits.shape[1])  # 计算平均对数概率
            else:  # 如果只有正样本
                # only pos  # 只有正样本
                mean_log_prob = -(logits * pos_mask).sum() / pos_mask.sum()  # 计算正样本的平均对数概率
        else:  # 如果是自对比
            # self contrast: push away from each other in the same time slice.  # 自对比：在同一时间切片中相互推开
            exp_logits = torch.exp(logits) * (1 - pos_mask)  # 计算指数logits并排除正样本
            sum_exp_logits = exp_logits.sum(1)  # 计算指数logits的和
            log_prob = -torch.log(sum_exp_logits + 1e-12)  # 计算对数概率

            mean_log_prob = -log_prob.sum() / (1 - pos_mask).sum()  # 计算平均对数概率

        return mean_log_prob  # 返回平均对数概率
