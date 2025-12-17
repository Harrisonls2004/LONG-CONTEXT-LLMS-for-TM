'''
This script is partially based on https://github.com/dallascard/scholar.  # 此脚本部分基于scholar项目
'''

import os  # 导入操作系统模块
import re  # 导入正则表达式模块
import string  # 导入字符串模块
import gensim.downloader  # 导入gensim下载器
from collections import Counter  # 导入计数器
import numpy as np  # 导入numpy库
import scipy.sparse  # 导入scipy稀疏矩阵模块
from tqdm import tqdm  # 导入进度条
from sklearn.feature_extraction.text import CountVectorizer  # 导入文本特征提取器

from topmost.data import file_utils  # 导入文件工具模块
from topmost.utils._utils import get_stopwords_set  # 导入获取停用词集合函数
from topmost.utils.logger import Logger  # 导入日志记录器


logger = Logger("WARNING")  # 创建警告级别的日志记录器


# compile some regexes  # 编译一些正则表达式
punct_chars = list(set(string.punctuation) - set("'"))  # 获取除单引号外的标点符号
punct_chars.sort()  # 排序标点符号
punctuation = ''.join(punct_chars)  # 连接标点符号
replace = re.compile('[%s]' % re.escape(punctuation))  # 编译标点符号替换正则表达式
alpha = re.compile('^[a-zA-Z_]+$')  # 编译纯字母正则表达式
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')  # 编译字母或数字正则表达式
alphanum = re.compile('^[a-zA-Z0-9_]+$')  # 编译字母数字组合正则表达式


class Tokenizer:  # 定义分词器类
    def __init__(self,  # 初始化方法
                 stopwords="English",  # 停用词，默认英语
                 keep_num=False,  # 是否保留数字，默认否
                 keep_alphanum=False,  # 是否保留字母数字组合，默认否
                 strip_html=False,  # 是否去除HTML标签，默认否
                 no_lower=False,  # 是否不转换为小写，默认否
                 min_length=3,  # 最小长度，默认3
                ):
        self.keep_num = keep_num  # 设置是否保留数字
        self.keep_alphanum = keep_alphanum  # 设置是否保留字母数字组合
        self.strip_html = strip_html  # 设置是否去除HTML标签
        self.lower = not no_lower  # 设置是否转换为小写
        self.min_length = min_length  # 设置最小长度

        self.stopword_set = get_stopwords_set(stopwords)  # 获取停用词集合

    def clean_text(self, text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):  # 定义文本清理方法
        # remove html tags  # 去除HTML标签
        if strip_html:  # 如果需要去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
        else:  # 否则
            # replace angle brackets  # 替换尖括号
            text = re.sub(r'<', '(', text)  # 将<替换为(
            text = re.sub(r'>', ')', text)  # 将>替换为)
        # lower case  # 转换为小写
        if lower:  # 如果需要转换为小写
            text = text.lower()  # 转换为小写
        # eliminate email addresses  # 去除邮箱地址
        if not keep_emails:  # 如果不保留邮箱
            text = re.sub(r'\S+@\S+', ' ', text)  # 去除邮箱地址
        # eliminate @mentions  # 去除@提及
        if not keep_at_mentions:  # 如果不保留@提及
            text = re.sub(r'\s@\S+', ' ', text)  # 去除@提及
        # replace underscores with spaces  # 将下划线替换为空格
        text = re.sub(r'_', ' ', text)  # 将下划线替换为空格
        # break off single quotes at the ends of words  # 去除单词末尾的单引号
        text = re.sub(r'\s\'', ' ', text)  # 去除单词末尾的单引号
        text = re.sub(r'\'\s', ' ', text)  # 去除单词开头的单引号
        # remove periods  # 去除句号
        text = re.sub(r'\.', '', text)  # 去除句号
        # replace all other punctuation (except single quotes) with spaces  # 将所有其他标点符号替换为空格
        text = replace.sub(' ', text)  # 将标点符号替换为空格
        # remove single quotes  # 去除单引号
        text = re.sub(r'\'', '', text)  # 去除单引号
        # replace all whitespace with a single space  # 将所有空白字符替换为单个空格
        text = re.sub(r'\s', ' ', text)  # 将空白字符替换为空格
        # strip off spaces on either end  # 去除首尾空格
        text = text.strip()  # 去除首尾空格
        return text  # 返回清理后的文本

    def tokenize(self, text):  # 定义分词方法
        text = self.clean_text(text, self.strip_html, self.lower)  # 清理文本
        tokens = text.split()  # 按空格分词

        tokens = ['_' if t in self.stopword_set else t for t in tokens]  # 将停用词替换为下划线

        # remove tokens that contain numbers  # 去除包含数字的词
        if not self.keep_alphanum and not self.keep_num:  # 如果不保留字母数字组合且不保留数字
            tokens = [t if alpha.match(t) else '_' for t in tokens]  # 只保留纯字母词

        # or just remove tokens that contain a combination of letters and numbers  # 或者只去除包含字母和数字组合的词
        elif not self.keep_alphanum:  # 如果不保留字母数字组合
            tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]  # 只保留纯字母或纯数字词

        # drop short tokens  # 去除短词
        if self.min_length > 0:  # 如果最小长度大于0
            tokens = [t if len(t) >= self.min_length else '_' for t in tokens]  # 去除短于最小长度的词

        unigrams = [t for t in tokens if t != '_']  # 过滤掉下划线标记的词
        # counts = Counter()  # 计数器（注释掉）
        # counts.update(unigrams)  # 更新计数器（注释掉）

        return unigrams  # 返回一元词


def make_word_embeddings(vocab):  # 定义制作词嵌入函数
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')  # 加载GloVe词向量
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))  # 初始化词嵌入矩阵

    num_found = 0  # 初始化找到的词数

    try:  # 尝试获取关键词列表
        key_word_list = glove_vectors.index_to_key  # 获取关键词列表（新版本）
    except:  # 如果失败
        key_word_list = glove_vectors.index2word  # 获取关键词列表（旧版本）

    for i, word in enumerate(tqdm(vocab, desc="loading word embeddings")):  # 遍历词汇表
        if word in key_word_list:  # 如果词在GloVe中
            word_embeddings[i] = glove_vectors[word]  # 获取词向量
            num_found += 1  # 增加找到的词数

    logger.info(f'number of found embeddings: {num_found}/{len(vocab)}')  # 记录找到的词嵌入数量

    return scipy.sparse.csr_matrix(word_embeddings)  # 返回稀疏矩阵格式的词嵌入


class Preprocess:  # 定义预处理类
    def __init__(self,  # 初始化方法
                 tokenizer=None,  # 分词器，默认为None
                 test_sample_size=None,  # 测试集样本大小，默认为None
                 test_p=0.2,  # 测试集比例，默认0.2
                 stopwords="English",  # 停用词，默认英语
                 min_doc_count=0,  # 最小文档频率，默认0
                 max_doc_freq=1.0,  # 最大文档频率，默认1.0
                 keep_num=False,  # 是否保留数字，默认否
                 keep_alphanum=False,  # 是否保留字母数字组合，默认否
                 strip_html=False,  # 是否去除HTML标签，默认否
                 no_lower=False,  # 是否不转换为小写，默认否
                 min_length=3,  # 最小长度，默认3
                 min_term=0,  # 最小词数，默认0
                 vocab_size=None,  # 词汇表大小，默认为None
                 seed=42,  # 随机种子，默认42
                 verbose=True  # 详细输出，默认是
                ):
        """
        Args:  # 参数说明
            test_sample_size:  # 测试样本大小
                Size of the test set.  # 测试集的大小
            test_p:  # 测试集比例
                Proportion of the test set. This helps sample the train set based on the size of the test set.  # 测试集的比例，帮助基于测试集大小采样训练集
            stopwords:  # 停用词
                List of stopwords to exclude.  # 要排除的停用词列表
            min-doc-count:  # 最小文档计数
                Exclude words that occur in less than this number of documents.  # 排除在此数量文档中出现次数少于这个值的词
            max_doc_freq:  # 最大文档频率
                Exclude words that occur in more than this proportion of documents.  # 排除在此比例文档中出现次数多于这个值的词
            keep-num:  # 保留数字
                Keep tokens made of only numbers.  # 保留仅由数字组成的词
            keep-alphanum:  # 保留字母数字组合
                Keep tokens made of a mixture of letters and numbers.  # 保留字母和数字混合的词
            strip_html:  # 去除HTML
                Strip HTML tags.  # 去除HTML标签
            no-lower:  # 不转换为小写
                Do not lowercase text  # 不将文本转换为小写
            min_length:  # 最小长度
                Minimum token length.  # 最小词长度
            min_term:  # 最小词数
                Minimum term number  # 最小词数
            vocab-size:  # 词汇表大小
                Size of the vocabulary (by most common in the union of train and test sets, following above exclusions)  # 词汇表大小（按训练集和测试集并集中最常见词排序，遵循上述排除规则）
            seed:  # 随机种子
                Random integer seed (only relevant for choosing test set)  # 随机整数种子（仅与选择测试集相关）
        """

        self.test_sample_size = test_sample_size  # 设置测试样本大小
        self.min_doc_count = min_doc_count  # 设置最小文档计数
        self.max_doc_freq = max_doc_freq  # 设置最大文档频率
        self.min_term = min_term  # 设置最小词数
        self.test_p = test_p  # 设置测试集比例
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.seed = seed  # 设置随机种子

        if tokenizer is not None:  # 如果分词器不为空
            self.tokenizer = tokenizer  # 使用传入的分词器
        else:  # 否则
            self.tokenizer = Tokenizer(stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length).tokenize  # 创建默认分词器

        if verbose:  # 如果需要详细输出
            logger.set_level("DEBUG")  # 设置日志级别为调试
        else:  # 否则
            logger.set_level("WARNING")  # 设置日志级别为警告

    def parse(self, texts, vocab):  # 定义解析方法
        if not isinstance(texts, list):  # 如果文本不是列表
            texts = [texts]  # 转换为列表

        vocab_set = set(vocab)  # 将词汇表转换为集合
        parsed_texts = list()
        for i, text in enumerate(tqdm(texts, desc="parsing texts")):
            tokens = self.tokenizer(text)
            tokens = [t for t in tokens if t in vocab_set]
            parsed_texts.append(" ".join(tokens))

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        sparse_bow = vectorizer.fit_transform(parsed_texts)
        return parsed_texts, sparse_bow

    def preprocess_jsonlist(self, dataset_dir, label_name=None):
        train_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'train.jsonlist'))
        test_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'test.jsonlist'))

        logger.info(f"Found training documents {len(train_items)} testing documents {len(test_items)}")

        raw_train_texts = []
        train_labels = []
        raw_test_texts = []
        test_labels = []

        for item in train_items:
            raw_train_texts.append(item['text'])

            if label_name is not None:
                train_labels.append(item[label_name])
 
        for item in test_items:
            raw_test_texts.append(item['text'])

            if label_name is not None:
                test_labels.append(item[label_name])

        rst = self.preprocess(raw_train_texts, train_labels, raw_test_texts, test_labels)

        return rst

    def convert_labels(self, train_labels, test_labels):
        if train_labels is not None:
            label_list = list(set(train_labels).union(set(test_labels)))
            label_list.sort()
            n_labels = len(label_list)
            label2id = dict(zip(label_list, range(n_labels)))

            logger.info(f"label2id: {label2id}")

            train_labels = [label2id[label] for label in train_labels]

            if test_labels is not None:
                test_labels = [label2id[label] for label in test_labels]

        return train_labels, test_labels

    def preprocess(
            self,
            raw_train_texts,
            train_labels=None,
            raw_test_texts=None,
            test_labels=None,
            pretrained_WE=False
        ):
        np.random.seed(self.seed)

        train_texts = list()
        test_texts = list()
        word_counts = Counter()
        doc_counts_counter = Counter()

        train_labels, test_labels = self.convert_labels(train_labels, test_labels)

        for text in tqdm(raw_train_texts, desc="loading train texts"):
            tokens = self.tokenizer(text)
            word_counts.update(tokens)
            doc_counts_counter.update(set(tokens))
            parsed_text = ' '.join(tokens)
            train_texts.append(parsed_text)

        if raw_test_texts:
            for text in tqdm(raw_test_texts, desc="loading test texts"):
                tokens = self.tokenizer(text)
                word_counts.update(tokens)
                doc_counts_counter.update(set(tokens))
                parsed_text = ' '.join(tokens)
                test_texts.append(parsed_text)

        words, doc_counts = zip(*doc_counts_counter.most_common())
        doc_freqs = np.array(doc_counts) / float(len(train_texts) + len(test_texts))

        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]

        # filter vocabulary
        if self.vocab_size is not None:
            vocab = vocab[:self.vocab_size]

        vocab.sort()

        train_idx = [i for i, text in enumerate(train_texts) if len(text.split()) >= self.min_term]
        train_idx = np.asarray(train_idx)

        if raw_test_texts is not None:
            test_idx = [i for i, text in enumerate(test_texts) if len(text.split()) >= self.min_term]
            test_idx = np.asarray(test_idx)

        # randomly sample
        if self.test_sample_size:
            logger.info("sample train and test sets...")

            train_num = len(train_idx)
            test_num = len(test_idx)
            test_sample_size = min(test_num, self.test_sample_size)
            train_sample_size = int((test_sample_size / self.test_p) * (1 - self.test_p))
            if train_sample_size > train_num:
                test_sample_size = int((train_num / (1 - self.test_p)) * self.test_p)
                train_sample_size = train_num

            train_idx = train_idx[np.sort(np.random.choice(train_num, train_sample_size, replace=False))]
            test_idx = test_idx[np.sort(np.random.choice(test_num, test_sample_size, replace=False))]

            logger.info(f"sampled train size: {len(train_idx)}")
            logger.info(f"sampled train size: {len(test_idx)}")

        train_texts, train_bow = self.parse([train_texts[i] for i in train_idx], vocab)

        rst = {
            'vocab': vocab,
            'train_bow': train_bow,
            "train_texts": train_texts
        }

        if train_labels is not None:
            rst['train_labels'] = np.asarray(train_labels)[train_idx]

        logger.info(f"Real vocab size: {len(vocab)}")
        logger.info(f"Real training size: {len(train_texts)} \t avg length: {rst['train_bow'].sum() / len(train_texts):.3f}")

        if raw_test_texts is not None:
            rst['test_texts'], rst['test_bow'] = self.parse(np.asarray(test_texts)[test_idx].tolist(), vocab)

            if test_labels is not None:
                rst['test_labels'] = np.asarray(test_labels)[test_idx]

            logger.info(f"Real testing size: {len(rst['test_texts'])} \t avg length: {rst['test_bow'].sum() / len(rst['test_texts']):.3f}")

        if pretrained_WE:
            rst['word_embeddings'] = make_word_embeddings(vocab)

        return rst

    def save(self, output_dir, vocab, train_texts, train_bow, word_embeddings=None, train_labels=None, test_texts=None, test_bow=None, test_labels=None):
        file_utils.make_dir(output_dir)

        file_utils.save_text(vocab, f"{output_dir}/vocab.txt")
        file_utils.save_text(train_texts, f"{output_dir}/train_texts.txt")
        scipy.sparse.save_npz(f"{output_dir}/train_bow.npz", scipy.sparse.csr_matrix(train_bow))
        if word_embeddings:
            scipy.sparse.save_npz(f"{output_dir}/word_embeddings.npz", word_embeddings)

        if train_labels is not None:
            np.savetxt(f"{output_dir}/train_labels.txt", train_labels, fmt='%i')

        if test_bow is not None:
            scipy.sparse.save_npz(f"{output_dir}/test_bow.npz", scipy.sparse.csr_matrix(test_bow))

        if test_texts is not None:
            file_utils.save_text(test_texts, f"{output_dir}/test_texts.txt")

            if test_labels is not None:
                np.savetxt(f"{output_dir}/test_labels.txt", test_labels, fmt='%i')
