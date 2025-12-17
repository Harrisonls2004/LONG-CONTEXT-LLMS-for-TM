import torch  # 导入PyTorch库
from torch.utils.data import DataLoader  # 导入数据加载器
import numpy as np  # 导入numpy库
import scipy.io  # 导入scipy输入输出模块
import scipy.sparse  # 导入scipy稀疏矩阵模块
from scipy.sparse import issparse  # 导入稀疏矩阵检查函数
from sentence_transformers import SentenceTransformer  # 导入句子转换器
from topmost.preprocess import Preprocess  # 导入预处理模块
from . import file_utils  # 导入文件工具模块
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable  # 导入类型提示


class DocEmbedModel:  # 定义文档嵌入模型类
    def __init__(  # 初始化方法
            self,
            model: Union[str, callable]="all-MiniLM-L6-v2",  # 模型参数，默认使用all-MiniLM-L6-v2
            device: str='cpu',  # 设备参数，默认使用CPU
            verbose: bool=False  # 详细输出参数，默认关闭
        ):
        self.verbose = verbose  # 设置详细输出标志

        if isinstance(model, str):  # 如果模型是字符串类型
            self.model = SentenceTransformer(model, device=device)  # 创建句子转换器模型
        else:  # 否则
            self.model = model  # 直接使用传入的模型

    def encode(self,  # 编码方法
               docs:List[str],  # 文档列表
               convert_to_tensor: bool=False  # 是否转换为张量，默认否
            ):

        embeddings = self.model.encode(  # 使用模型编码文档
                        docs,  # 文档列表
                        convert_to_tensor=convert_to_tensor,  # 是否转换为张量
                        show_progress_bar=self.verbose  # 是否显示进度条
                    )
        return embeddings  # 返回嵌入结果


class RawDataset:  # 定义原始数据集类
    def __init__(self,  # 初始化方法
                 docs,  # 文档列表
                 preprocess=None,  # 预处理对象，默认为None
                 batch_size=200,  # 批次大小，默认200
                 device='cpu',  # 设备，默认CPU
                 as_tensor=True,  # 是否转换为张量，默认是
                 contextual_embed=False,  # 是否使用上下文嵌入，默认否
                 pretrained_WE=False,  # 是否使用预训练词嵌入，默认否
                 doc_embed_model="all-MiniLM-L6-v2",  # 文档嵌入模型，默认all-MiniLM-L6-v2
                 embed_model_device=None,  # 嵌入模型设备，默认为None
                 verbose=False  # 详细输出，默认关闭
                ):

        if preprocess is None:  # 如果没有预处理对象
            preprocess = Preprocess(verbose=verbose)  # 创建预处理对象

        rst = preprocess.preprocess(docs, pretrained_WE=pretrained_WE)  # 预处理文档
        self.train_data = rst["train_bow"]  # 获取训练词袋数据
        self.train_texts = rst["train_texts"]  # 获取训练文本
        self.vocab = rst["vocab"]  # 获取词汇表
        if issparse(self.train_data):  # 如果是稀疏矩阵
            self.train_data = self.train_data.toarray()  # 转换为密集矩阵

        self.vocab_size = len(self.vocab)  # 计算词汇表大小

        if contextual_embed:  # 如果需要上下文嵌入
            if embed_model_device is None:  # 如果嵌入模型设备未指定
                embed_model_device = device  # 使用默认设备

            if isinstance(doc_embed_model, str):  # 如果文档嵌入模型是字符串
                self.doc_embedder = DocEmbedModel(doc_embed_model, embed_model_device, verbose=verbose)  # 创建文档嵌入模型
            else:  # 否则
                self.doc_embedder = doc_embed_model  # 直接使用传入的模型

            self.train_contextual_embed = self.doc_embedder.encode(docs)  # 编码文档获取上下文嵌入
            self.contextual_embed_size = self.train_contextual_embed.shape[1]  # 获取上下文嵌入维度

        if as_tensor:  # 如果需要转换为张量
            if contextual_embed:  # 如果有上下文嵌入
                self.train_data = np.concatenate((self.train_data, self.train_contextual_embed), axis=1)  # 连接词袋数据和上下文嵌入

            self.train_data = torch.from_numpy(self.train_data).float().to(device)  # 转换为PyTorch张量
            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)  # 创建数据加载器


class BasicDataset:  # 定义基础数据集类
    def __init__(self,  # 初始化方法
                 dataset_dir,  # 数据集目录
                 batch_size=200,  # 批次大小，默认200
                 read_labels=False,  # 是否读取标签，默认否
                 as_tensor=True,  # 是否转换为张量，默认是
                 contextual_embed=False,  # 是否使用上下文嵌入，默认否
                 doc_embed_model="all-MiniLM-L6-v2",  # 文档嵌入模型，默认all-MiniLM-L6-v2
                 device='cpu'  # 设备，默认CPU
                ):
        # train_bow: NxV  # 训练词袋矩阵：文档数x词汇表大小
        # test_bow: Nxv  # 测试词袋矩阵：文档数x词汇表大小
        # word_emeddings: VxD  # 词嵌入矩阵：词汇表大小x嵌入维度
        # vocab: V, ordered by word id.  # 词汇表：按词ID排序

        self.load_data(dataset_dir, read_labels)  # 加载数据
        self.vocab_size = len(self.vocab)  # 计算词汇表大小

        print("train_size: ", self.train_bow.shape[0])  # 打印训练集大小
        print("test_size: ", self.test_bow.shape[0])  # 打印测试集大小
        print("vocab_size: ", self.vocab_size)  # 打印词汇表大小
        print("average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))  # 打印平均文档长度

        if contextual_embed:  # 如果需要上下文嵌入
            self.doc_embedder = DocEmbedModel(doc_embed_model, device)  # 创建文档嵌入模型
            self.train_contextual_embed = self.doc_embedder.encode(self.train_texts)  # 编码训练文本
            self.test_contextual_embed = self.doc_embedder.encode(self.test_texts)  # 编码测试文本

            self.contextual_embed_size = self.train_contextual_embed.shape[1]  # 获取上下文嵌入维度

        if as_tensor:  # 如果需要转换为张量
            if not contextual_embed:  # 如果没有上下文嵌入
                self.train_data = self.train_bow  # 训练数据为词袋数据
                self.test_data = self.test_bow  # 测试数据为词袋数据
            else:  # 否则
                self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)  # 连接训练词袋和上下文嵌入
                self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)  # 连接测试词袋和上下文嵌入

            self.train_data = torch.from_numpy(self.train_data).to(device)  # 转换为训练张量
            self.test_data = torch.from_numpy(self.test_data).to(device)  # 转换为测试张量

            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)  # 创建训练数据加载器
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)  # 创建测试数据加载器

    def load_data(self, path, read_labels):  # 加载数据方法

        self.train_bow = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')  # 加载训练词袋数据
        self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')  # 加载测试词袋数据
        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')  # 加载预训练词嵌入

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')  # 读取训练文本
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')  # 读取测试文本

        if read_labels:  # 如果需要读取标签
            self.train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)  # 加载训练标签
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)  # 加载测试标签

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')  # 读取词汇表
