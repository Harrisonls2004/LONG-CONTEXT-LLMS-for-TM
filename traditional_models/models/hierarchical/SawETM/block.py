import torch.nn as nn  # 导入神经网络模块


def _get_activation_fn(activation):  # 定义获取激活函数的辅助函数
    if activation == "relu":  # 如果激活函数是ReLU
        return nn.ReLU()  # 返回ReLU激活函数
    elif activation == "softplus":  # 如果激活函数是Softplus
        return nn.Softplus()  # 返回Softplus激活函数
    elif activation == "tanh":  # 如果激活函数是Tanh
        return nn.Tanh()  # 返回Tanh激活函数
    else:  # 否则
        raise RuntimeError("activation should be relu/tanh/softplus, not {}".format(activation))  # 抛出运行时错误


class ResBlock(nn.Module):  # 定义残差块类
    """Simple MLP block with residual connection.  # 带残差连接的简单MLP块

    Args:  # 参数说明
        in_features: the feature dimension of each output sample.  # 输入特征维度
        out_features: the feature dimension of each output sample.  # 输出特征维度
        activation: the activation function of intermediate layer, relu or gelu.  # 中间层的激活函数，relu或gelu
    """

    def __init__(self, in_features, out_features, activation="relu"):  # 初始化方法
        super(ResBlock, self).__init__()  # 调用父类初始化
        self.in_features = in_features  # 设置输入特征维度
        self.out_features = out_features  # 设置输出特征维度
        self.fc1 = nn.Linear(in_features, out_features)  # 第一个全连接层
        self.fc2 = nn.Linear(out_features, out_features)  # 第二个全连接层

        self.bn = nn.BatchNorm1d(out_features)  # 批归一化层
        self.activation = _get_activation_fn(activation)  # 获取激活函数

    def forward(self, x):  # 定义前向传播方法
        if self.in_features == self.out_features:  # 如果输入输出维度相同
            out = self.fc2(self.activation(self.fc1(x)))  # 通过两个全连接层和激活函数
            return self.activation(self.bn(x + out))  # 残差连接后进行批归一化和激活
        else:  # 如果输入输出维度不同
            x = self.fc1(x)  # 先通过第一个全连接层调整维度
            out = self.fc2(self.activation(x))  # 通过激活函数和第二个全连接层
            return self.activation(self.bn(x + out))  # 残差连接后进行批归一化和激活
