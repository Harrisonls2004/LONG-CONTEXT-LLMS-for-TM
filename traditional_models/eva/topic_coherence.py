import numpy as np  # 导入numpy库
from tqdm import tqdm  # 导入进度条
from gensim.corpora import Dictionary  # 导入词典
from gensim.models import CoherenceModel  # 导入一致性模型
from ..data.file_utils import split_text_word  # 导入文本分词工具
from typing import List  # 导入类型提示


def _coherence(  # 定义主题一致性函数
        reference_corpus: List[str],  # 参考语料库
        vocab: List[str],  # 词汇表
        top_words: List[str],  # 主题词列表
        coherence_type='c_v',  # 一致性类型，默认c_v
        topn=20  # 前n个词，默认20
    ):
    split_top_words = split_text_word(top_words)  # 分词主题词
    split_reference_corpus = split_text_word(reference_corpus)  # 分词参考语料库
    dictionary = Dictionary(split_text_word(vocab))  # 创建词典

    cm = CoherenceModel(  # 创建一致性模型
        texts=split_reference_corpus,  # 文本
        dictionary=dictionary,  # 词典
        topics=split_top_words,  # 主题
        topn=topn,  # 前n个词
        coherence=coherence_type,  # 一致性类型
    )
    cv_per_topic = cm.get_coherence_per_topic()  # 获取每个主题的一致性
    score = np.mean(cv_per_topic)  # 计算平均一致性分数

    return score  # 返回一致性分数


def dynamic_coherence(train_texts, train_times, vocab, top_words_list, coherence_type='c_v', verbose=False):  # 定义动态一致性函数
    cv_score_list = list()  # 初始化一致性分数列表

    for time, top_words in tqdm(enumerate(top_words_list)):  # 遍历每个时间点的主题词
        # use the texts of each time slice as the reference corpus.  # 使用每个时间切片的文本作为参考语料库
        idx = np.where(train_times == time)[0]  # 获取该时间点的索引
        reference_corpus = [train_texts[i] for i in idx]  # 获取该时间点的参考语料库

        # use the topics at a time slice  # 使用该时间切片的主题
        cv_score = _coherence(reference_corpus, vocab, top_words, coherence_type)  # 计算一致性分数
        cv_score_list.append(cv_score)  # 添加到列表

    if verbose:  # 如果需要详细输出
        print(f"dynamic TC list: {cv_score_list}")  # 打印动态一致性列表

    return np.mean(cv_score_list)  # 返回平均动态一致性分数
