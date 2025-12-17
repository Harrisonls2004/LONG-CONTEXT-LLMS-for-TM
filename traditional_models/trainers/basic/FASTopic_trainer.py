from topmost.utils.logger import Logger  # 导入日志记录器
from topmost.preprocess import Preprocess  # 导入预处理模块
from fastopic import FASTopic  # 导入FASTopic库

logger = Logger("WARNING")  # 创建警告级别的日志记录器


class FASTopicTrainer:  # 定义FASTopic训练器类
    def __init__(self,  # 初始化方法
                 dataset,  # 数据集
                 num_topics=50,  # 主题数量，默认50
                 num_top_words=15,  # 主题词数量，默认15
                 preprocess=None,  # 预处理对象，默认None
                 epochs=200,  # 训练轮数，默认200
                 DT_alpha=3.0,  # 文档-主题alpha参数，默认3.0
                 TW_alpha=2.0,  # 主题-词alpha参数，默认2.0
                 theta_temp=1.0,  # 主题分布温度参数，默认1.0
                 verbose=False  # 详细输出，默认关闭
                ):
        self.dataset = dataset  # 设置数据集
        self.num_top_words = num_top_words  # 设置主题词数量

        preprocess = Preprocess(stopwords=[]) if preprocess is None else preprocess  # 如果没有预处理对象则创建默认的
        self.model = FASTopic(num_topics=num_topics,  # 创建FASTopic模型
                              preprocess=preprocess,  # 预处理对象
                              num_top_words=num_top_words,  # 主题词数量
                              DT_alpha=DT_alpha,  # 文档-主题alpha参数
                              TW_alpha=TW_alpha,  # 主题-词alpha参数
                              theta_temp=theta_temp,  # 主题分布温度参数
                              verbose=verbose  # 详细输出
                            )

        self.epochs = epochs  # 设置训练轮数

        if verbose:  # 如果需要详细输出
            logger.set_level("DEBUG")  # 设置日志级别为DEBUG
        else:  # 否则
            logger.set_level("WARNING")  # 设置日志级别为WARNING

    def train(self):  # 定义训练方法
        return self.model.fit_transform(self.dataset.train_texts, epochs=self.epochs)  # 训练模型并转换训练文本

    def test(self, texts):  # 定义测试方法
        theta = self.model.transform(texts)  # 转换文本获取主题分布
        return theta  # 返回主题分布

    def get_beta(self):  # 定义获取主题-词分布方法
        beta = self.model.get_beta()  # 获取主题-词分布
        return beta  # 返回主题-词分布

    def get_top_words(self, num_top_words=None):  # 定义获取主题词方法
        if num_top_words is None:  # 如果没有指定主题词数量
            num_top_words = self.num_top_words  # 使用默认的主题词数量
        return self.model.get_top_words(num_top_words)  # 返回主题词

    def export_theta(self):  # 定义导出主题分布方法
        train_theta = self.test(self.dataset.train_texts)  # 获取训练集主题分布
        test_theta = self.test(self.dataset.test_texts)  # 获取测试集主题分布
        return train_theta, test_theta  # 返回训练和测试主题分布
