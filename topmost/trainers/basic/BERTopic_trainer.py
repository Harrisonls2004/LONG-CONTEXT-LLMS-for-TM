class BERTopicTrainer:  # 定义BERTopic训练器类
    def __init__(self, dataset, num_topics=50, num_top_words=15):  # 初始化方法
        from bertopic import BERTopic  # 导入BERTopic库
        self.model = BERTopic(nr_topics=num_topics, top_n_words=num_top_words)  # 创建BERTopic模型
        self.dataset = dataset  # 设置数据集

    def train(self):  # 定义训练方法
        self.model.fit_transform(self.dataset.train_texts)  # 训练模型并转换训练文本
        top_words = self.get_top_words()  # 获取主题词
        train_theta = self.test(self.dataset.train_texts)  # 获取训练集的主题分布

        return top_words, train_theta  # 返回主题词和训练主题分布

    def test(self, texts):  # 定义测试方法
        theta, _ = self.model.approximate_distribution(texts)  # 近似计算文档的主题分布
        return theta  # 返回主题分布

    def get_beta(self):  # 定义获取主题-词分布方法
        # NOTE: beta is modeled as unnormalized c-tf_idf.  # 注意：beta建模为未归一化的c-tf_idf
        beta = self.model.c_tf_idf_.toarray()  # 获取c-tf_idf矩阵并转换为密集数组
        return beta  # 返回主题-词分布

    def get_top_words(self):  # 定义获取主题词方法
        top_words = list()  # 初始化主题词列表
        for item in self.model.get_topics().values():  # 遍历每个主题
            top_words.append(' '.join([x[0] for x in item]))  # 提取主题词并用空格连接
        return top_words  # 返回主题词列表

    def export_theta(self):  # 定义导出主题分布方法
        train_theta = self.test(self.dataset.train_texts)  # 获取训练集主题分布
        test_theta = self.test(self.dataset.test_texts)  # 获取测试集主题分布
        return train_theta, test_theta  # 返回训练和测试主题分布
