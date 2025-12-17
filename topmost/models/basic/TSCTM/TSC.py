import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


# Topic-Semantic Contrastive Learning  # 主题-语义对比学习
class TSC(nn.Module):  # 定义主题语义对比学习类
    def __init__(self, temperature=0.07, weight_contrast=None, use_aug=False):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.use_aug = use_aug  # 设置是否使用数据增强
        self.temperature = temperature  # 设置温度参数
        self.weight_contrast = weight_contrast  # 设置对比学习权重

    def forward(self, features, quant_idx=None, weight_same_quant=None):  # 定义前向传播方法
        device = features.device  # 获取张量所在设备

        batch_size = features.shape[0]  # 获取批次大小
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 创建单位矩阵掩码

        contrast_count = features.shape[1]  # 获取对比特征数量
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将特征展开并连接
        anchor_feature = contrast_feature  # 设置锚点特征
        anchor_count = contrast_count  # 设置锚点数量

        anchor_dot_contrast = torch.div(  # 计算锚点与对比特征的点积
            torch.matmul(anchor_feature, contrast_feature.T),  # 矩阵乘法
            self.temperature  # 除以温度参数
        )

        # for numerical stability  # 为了数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 找到最大值
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值以提高数值稳定性

        # tile mask  # 平铺掩码
        mask = mask.repeat(anchor_count, contrast_count)  # 重复掩码
        # mask-out self-contrast cases.  # 掩盖自对比情况
        # logits_mask is 1 - eye matrix  # logits_mask是1减去单位矩阵
        logits_mask = torch.scatter(  # 创建logits掩码
            torch.ones_like(mask),  # 全1矩阵
            1,  # 维度1
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # 索引
            0  # 设置为0
        )

        mask = mask * logits_mask  # 应用logits掩码

        t_quant_idx = quant_idx.contiguous().view(-1, 1)  # 重塑量化索引

        # quant_idx_mask: 1 means same quantization; 0 means different quantization  # 量化索引掩码：1表示相同量化，0表示不同量化
        quant_idx_mask = torch.eq(t_quant_idx, t_quant_idx.T).float()  # 创建量化索引掩码
        quant_idx_mask = quant_idx_mask.repeat(anchor_count, contrast_count)  # 重复量化索引掩码

        exp_logits = torch.exp(logits) * (1 - quant_idx_mask)  # 计算指数logits，排除相同量化的情况
        sum_exp_logits = exp_logits.sum(1, keepdim=True)  # 计算指数logits的和

        if not self.use_aug:  # 如果不使用数据增强
            # quant_idx_mask includes self-contrast cases.  # 量化索引掩码包括自对比情况
            # logits * logits_mask is to remove the positive pair but keep the negative pair in the self-contrast cases.  # logits * logits_mask用于移除正样本对但保留自对比情况下的负样本对
            # This is because some samples do not have positive pairs.  # 这是因为一些样本没有正样本对
            log_prob = logits * logits_mask - torch.log(sum_exp_logits + 1e-10)  # 计算对数概率
            mean_log_prob_pos = (quant_idx_mask * log_prob).sum(1) / quant_idx_mask.sum(1)  # 计算正样本的平均对数概率
        else:  # 如果使用数据增强
            log_prob = logits - torch.log(sum_exp_logits + 1e-10)  # 计算对数概率
            # between original and augmented samples.  # 原始样本和增强样本之间
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 计算正样本的平均对数概率

            # between original samples.  # 原始样本之间
            same_quant_mask = quant_idx_mask * logits_mask  # 相同量化掩码
            same_quant_mean_log_prob_pos = (same_quant_mask * log_prob).sum(1) / (same_quant_mask.sum(1) + 1e-10)  # 计算相同量化的平均对数概率
            mean_log_prob_pos += weight_same_quant * same_quant_mean_log_prob_pos  # 加权相同量化的平均对数概率

        loss = - self.weight_contrast * mean_log_prob_pos  # 计算对比损失
        loss = loss.view(anchor_count, batch_size).sum(axis=0).mean()  # 重塑并计算平均损失

        return loss  # 返回损失
