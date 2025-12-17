import gensim  # 导入gensim库
from gensim.models import LdaModel  # 导入LDA模型
from topmost.utils import _utils  # 导入工具函数
from topmost.utils.logger import Logger  # 导入日志记录器


logger = Logger("WARNING")  # 创建警告级别的日志记录器


class LDAGensimTrainer:  # 定义Gensim LDA训练器类
    def __init__(self,  # 初始化方法
                 dataset,  # 数据集
                 num_topics=50,  # 主题数量，默认50
                 num_top_words=15,  # 主题词数量，默认15
                 max_iter=1,  # 最大迭代次数，默认1
                 alpha="symmetric",  # alpha参数，默认对称
                 eta=None,  # eta参数，默认None
                 verbose=False  # 详细输出，默认关闭
                ):

        self.dataset = dataset  # 设置数据集
        self.num_topics = num_topics  # 设置主题数量
        self.vocab_size = dataset.vocab_size  # 设置词汇表大小
        self.max_iter = max_iter  # 设置最大迭代次数
        self.alpha = alpha  # 设置alpha参数
        self.eta = eta  # 设置eta参数
        self.verbose = verbose  # 设置详细输出
        self.num_top_words = num_top_words  # 设置主题词数量

    def train(self):  # 定义训练方法
        train_bow = self.dataset.train_bow.astype("int32")  # 将训练词袋数据转换为int32类型
        id2word = dict(zip(range(self.vocab_size), self.dataset.vocab))  # 创建ID到词的映射字典
        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)  # 将密集矩阵转换为gensim语料库格式
        self.model = LdaModel(  # 创建LDA模型
            corpus=corpus,  # 语料库
            id2word=id2word,  # ID到词的映射
            num_topics=self.num_topics,  # 主题数量
            passes=self.max_iter,  # 迭代次数
            alpha=self.alpha,  # alpha参数
            eta=self.eta  # eta参数
        )

        top_words = self.get_top_words()  # 获取主题词
        train_theta = self.test(self.dataset.train_bow)  # 获取训练集主题分布
        return top_words, train_theta  # 返回主题词和训练主题分布

    def test(self, bow):  # 定义测试方法
        bow = bow.astype('int64')  # 将词袋数据转换为int64类型
        corpus = gensim.matutils.Dense2Corpus(bow, documents_columns=False)  # 转换为gensim语料库格式
        theta = gensim.matutils.corpus2dense(self.model.get_document_topics(corpus), num_docs=bow.shape[0], num_terms=self.num_topics)  # 获取文档主题分布并转换为密集矩阵
        theta = theta.transpose()  # 转置矩阵
        return theta  # 返回主题分布

    def get_beta(self):  # 定义获取主题-词分布方法
        return self.model.get_topics()  # 返回主题-词分布

    def get_top_words(self, num_top_words=None):  # 定义获取主题词方法
        if num_top_words is None:  # 如果没有指定主题词数量
            num_top_words = self.num_top_words  # 使用默认的主题词数量
        beta = self.get_beta()  # 获取主题-词分布
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words)  # 获取主题词
        return top_words  # 返回主题词

    def export_theta(self):  # 定义导出主题分布方法
        train_theta = self.test(self.dataset.train_bow)  # 获取训练集主题分布
        test_theta = self.test(self.dataset.test_bow)  # 获取测试集主题分布
        return train_theta, test_theta  # 返回训练和测试主题分布


class LDASklearnTrainer:  # 定义Sklearn LDA训练器类
    def __init__(self,  # 初始化方法
                 model,  # 模型
                 dataset,  # 数据集
                 num_top_words=15,  # 主题词数量，默认15
                 verbose=False):  # 详细输出，默认关闭
        self.model = model  # 设置模型
        self.dataset = dataset  # 设置数据集
        self.num_top_words = num_top_words  # 设置主题词数量
        self.verbose = verbose  # 设置详细输出

    def train(self):  # 定义训练方法
        train_bow = self.dataset.train_bow.astype('int64')  # 将训练词袋数据转换为int64类型
        self.model.fit(train_bow)  # 训练模型

        top_words = self.get_top_words()  # 获取主题词
        train_theta = self.test(self.dataset.train_bow)  # 获取训练集主题分布

        return top_words, train_theta  # 返回主题词和训练主题分布

    def test(self, bow):  # 定义测试方法
        bow = bow.astype('int64')  # 将词袋数据转换为int64类型
        return self.model.transform(bow.astype('int64'))  # 转换数据获取主题分布

    def get_beta(self):  # 定义获取主题-词分布方法
        return self.model.components_  # 返回模型的主题-词分布

    def get_top_words(self, num_top_words=None):  # 定义获取主题词方法
        if num_top_words is None:  # 如果没有指定主题词数量
            num_top_words = self.num_top_words  # 使用默认的主题词数量

        beta = self.get_beta()  # 获取主题-词分布
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words, self.verbose)  # 获取主题词
        return top_words  # 返回主题词

    def export_theta(self):  # 定义导出主题分布方法
        train_theta = self.test(self.dataset.train_bow)  # 获取训练集主题分布
        test_theta = self.test(self.dataset.test_bow)  # 获取测试集主题分布
        return train_theta, test_theta  # 返回训练和测试主题分布
