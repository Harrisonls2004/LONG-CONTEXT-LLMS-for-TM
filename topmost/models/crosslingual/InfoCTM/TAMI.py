import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口


class TAMI(nn.Module):  # 定义主题对齐互信息类
    '''
        InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling. AAAI 2023
        InfoCTM：跨语言主题建模的互信息最大化视角，AAAI 2023

        Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu  # 作者信息
    '''
    def __init__(self, temperature, weight_MI, pos_threshold, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn):  # 初始化方法
        super().__init__()  # 调用父类初始化
        self.temperature = temperature  # 设置温度参数
        self.weight_MI = weight_MI  # 设置互信息权重
        self.pos_threshold = pos_threshold  # 设置正样本阈值
        self.pretrain_word_embeddings_en = pretrain_word_embeddings_en  # 设置英文预训练词嵌入
        self.pretrain_word_embeddings_cn = pretrain_word_embeddings_cn  # 设置中文预训练词嵌入

        self.trans_e2c = torch.as_tensor(trans_e2c).float()  # 将英文到中文翻译矩阵转换为张量
        self.trans_e2c = nn.Parameter(self.trans_e2c, requires_grad=False)  # 设置为不可训练参数
        self.trans_c2e = self.trans_e2c.T  # 中文到英文翻译矩阵为转置

        pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn = self.compute_pos_neg(pretrain_word_embeddings_en, pretrain_word_embeddings_cn, self.trans_e2c, self.trans_c2e)  # 计算正负样本掩码
        self.pos_trans_mask_en = nn.Parameter(pos_trans_mask_en, requires_grad=False)  # 英文正样本翻译掩码
        self.pos_trans_mask_cn = nn.Parameter(pos_trans_mask_cn, requires_grad=False)  # 中文正样本翻译掩码
        self.neg_trans_mask_en = nn.Parameter(neg_trans_mask_en, requires_grad=False)  # 英文负样本翻译掩码
        self.neg_trans_mask_cn = nn.Parameter(neg_trans_mask_cn, requires_grad=False)  # 中文负样本翻译掩码

    def build_CVL_mask(self, embeddings):  # 定义构建跨语言视觉掩码方法
        norm_embed = F.normalize(embeddings)  # 归一化嵌入
        cos_sim = torch.matmul(norm_embed, norm_embed.T)  # 计算余弦相似度
        pos_mask = (cos_sim >= self.pos_threshold).float()  # 创建正样本掩码
        return pos_mask  # 返回正样本掩码

    def translation_mask(self, mask, trans_dict_matrix):  # 定义翻译掩码方法
        # V1 x V2  # 词汇表1大小 x 词汇表2大小
        trans_mask = torch.matmul(mask, trans_dict_matrix)  # 计算翻译掩码
        return trans_mask  # 返回翻译掩码

    def compute_pos_neg(self, pretrain_word_embeddings_en, pretrain_word_embeddings_cn, trans_e2c, trans_c2e):  # 定义计算正负样本方法
        # Ve x Ve  # 英文词汇表大小 x 英文词汇表大小
        pos_mono_mask_en = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_en))  # 构建英文单语正样本掩码
        # Vc x Vc  # 中文词汇表大小 x 中文词汇表大小
        pos_mono_mask_cn = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_cn))  # 构建中文单语正样本掩码

        # Ve x Vc  # 英文词汇表大小 x 中文词汇表大小
        pos_trans_mask_en = self.translation_mask(pos_mono_mask_en, trans_e2c)  # 构建英文翻译正样本掩码
        pos_trans_mask_cn = self.translation_mask(pos_mono_mask_cn, trans_c2e)  # 构建中文翻译正样本掩码

        neg_trans_mask_en = (pos_trans_mask_en <= 0).float()  # 构建英文翻译负样本掩码
        neg_trans_mask_cn = (pos_trans_mask_cn <= 0).float()  # 构建中文翻译负样本掩码

        return pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn  # 返回正负样本掩码

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask):  # 定义互信息计算方法
        anchor_dot_contrast = torch.div(  # 计算锚点与对比特征的点积
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),  # 归一化后的矩阵乘法
            self.temperature  # 除以温度参数
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 找到最大值
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值以提高数值稳定性

        exp_logits = torch.exp(logits) * neg_mask  # 计算指数logits并应用负样本掩码
        sum_exp_logits = exp_logits.sum(1, keepdim=True)  # 计算指数logits的和

        log_prob = logits - torch.log(sum_exp_logits + torch.exp(logits) + 1e-10)  # 计算对数概率
        mean_log_prob = -(mask * log_prob).sum()  # 计算平均对数概率
        return mean_log_prob  # 返回平均对数概率

    def forward(self, fea_en, fea_cn):  # 定义前向传播方法
        loss_TAMI = self.MutualInfo(fea_en, fea_cn, self.pos_trans_mask_en, self.neg_trans_mask_en)  # 计算英文到中文的互信息损失
        loss_TAMI += self.MutualInfo(fea_cn, fea_en, self.pos_trans_mask_cn, self.neg_trans_mask_cn)  # 加上中文到英文的互信息损失

        loss_TAMI = loss_TAMI / (self.pos_trans_mask_en.sum() + self.pos_trans_mask_cn.sum())  # 归一化损失

        loss_TAMI = self.weight_MI * loss_TAMI  # 乘以权重
        return loss_TAMI  # 返回TAMI损失
