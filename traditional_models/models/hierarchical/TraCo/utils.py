import torch  # 导入PyTorch库


def get_topk_tensor(matrix, topk, return_mask=False):  # 定义获取top-k张量的函数
    topk_values, topk_idx = torch.topk(matrix, topk)  # 获取top-k值和索引
    topk_tensor = torch.zeros_like(matrix)  # 创建与原矩阵相同形状的零张量
    if return_mask:  # 如果返回掩码
        topk_tensor.scatter_(1, topk_idx, 1)  # 在top-k位置填入1
    else:  # 否则
        topk_tensor.scatter_(1, topk_idx, topk_values)  # 在top-k位置填入对应的值

    return topk_tensor  # 返回top-k张量


def pairwise_euclidean_distance(x, y):  # 定义计算成对欧几里得距离的函数
    cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, dim=-1) - 2 * torch.matmul(x, y.t())  # 计算欧几里得距离的平方
    return cost  # 返回距离成本矩阵
