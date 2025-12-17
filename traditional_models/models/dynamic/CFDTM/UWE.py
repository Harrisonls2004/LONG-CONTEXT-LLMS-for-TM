import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class UWE(nn.Module):  # 定义未观察词嵌入类
    def __init__(self, ETC, num_times, temperature, weight_UWE, neg_topk):  # 初始化方法
        super().__init__()  # 调用父类初始化

        self.ETC = ETC  # 设置嵌入时间对比学习模块
        self.weight_UWE = weight_UWE  # 设置UWE损失权重
        self.num_times = num_times  # 设置时间步数
        self.temperature = temperature  # 设置温度参数
        self.neg_topk = neg_topk  # 设置负样本top-k数量

    def forward(self, time_wordcount, beta, topic_embeddings, word_embeddings):  # 定义前向传播方法
        assert(self.num_times == time_wordcount.shape[0])  # 断言时间步数一致

        topk_indices = self.get_topk_indices(beta)  # 获取top-k索引

        loss_UWE = 0.  # 初始化UWE损失
        cnt_valid_times = 0.  # 初始化有效时间计数
        for t in range(self.num_times):  # 遍历每个时间步
            neg_idx = torch.where(time_wordcount[t] == 0)[0]  # 找到该时间步中词频为0的词索引

            time_topk_indices = topk_indices[t]  # 获取该时间步的top-k索引
            neg_idx = list(set(neg_idx.cpu().tolist()).intersection(set(time_topk_indices.cpu().tolist())))  # 计算负样本索引的交集
            neg_idx = torch.tensor(neg_idx).long().to(time_wordcount.device)  # 转换为张量

            if len(neg_idx) == 0:  # 如果没有负样本
                continue  # 跳过当前时间步

            time_neg_WE = word_embeddings[neg_idx]  # 获取负样本词嵌入

            # topic_embeddings[t]: K x D  # 主题嵌入：主题数 x 嵌入维度
            # word_embeddings[neg_idx]: |V_{neg}| x D  # 负样本词嵌入：负样本数 x 嵌入维度
            loss_UWE += self.ETC.compute_loss(topic_embeddings[t], time_neg_WE, temperature=self.temperature, all_neg=True)  # 计算UWE损失
            cnt_valid_times += 1  # 有效时间计数加1

        if cnt_valid_times > 0:  # 如果有有效时间步
            loss_UWE *= (self.weight_UWE / cnt_valid_times)  # 归一化UWE损失

        return loss_UWE  # 返回UWE损失

    def get_topk_indices(self, beta):  # 定义获取top-k索引方法
        # topk_indices: T x K x neg_topk  # top-k索引：时间步数 x 主题数 x 负样本top-k数量
        topk_indices = torch.topk(beta, k=self.neg_topk, dim=-1).indices  # 获取top-k索引
        topk_indices = torch.flatten(topk_indices, start_dim=1)  # 展平索引
        return topk_indices  # 返回top-k索引
