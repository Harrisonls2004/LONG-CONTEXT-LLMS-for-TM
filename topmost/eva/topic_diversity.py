import numpy as np  # 导入numpy库
from collections import Counter  # 导入计数器
from tqdm import tqdm  # 导入进度条

from typing import List  # 导入类型提示


def _diversity(top_words: List[str]):  # 定义主题多样性函数
    num_words = 0.  # 初始化词数
    word_set = set()  # 初始化词集合
    for words in top_words:  # 遍历每个主题词列表
        ws = words.split()  # 分词
        num_words += len(ws)  # 累加词数
        word_set.update(ws)  # 更新词集合

    TD = len(word_set) / num_words  # 计算主题多样性
    return TD  # 返回主题多样性值


def multiaspect_diversity(top_words: List[str], _type="TD"):  # 定义多方面多样性函数
    TD_list = list()  # 初始化多样性列表
    for level_top_words in top_words:  # 遍历每个层次的主题词
        TD = _diversity(level_top_words, _type)  # 计算该层次的多样性
        TD_list.append(TD)  # 添加到列表

    return np.mean(TD_list)  # 返回平均多样性


def _time_slice_diversity(topics, time_vocab):  # 定义时间切片多样性函数
    num_associated_words = 0.  # 初始化关联词数
    T = len(topics[0].split())  # 获取主题词数量
    flatten_topic_words = [word for topic_words in topics for word in topic_words.split()]  # 展平主题词
    counter = Counter(flatten_topic_words)  # 统计词频

    # for word in np.sort(list(set(flatten_topic_words))):  # 注释掉的代码
    for word in np.sort(flatten_topic_words):  # 遍历排序后的词
        if (counter[word] == 1) and word in time_vocab:  # 如果词只出现一次且在时间词汇中
            num_associated_words += 1  # 增加关联词数

    return num_associated_words / (len(topics) * T)  # 返回时间切片多样性


def dynamic_diversity(  # 定义动态多样性函数
        top_words: List[str],  # 主题词列表
        train_bow: np.ndarray,  # 训练词袋矩阵
        train_times: List[int],  # 训练时间列表
        vocab: List[str],  # 词汇表
        verbose=False  # 详细输出标志
    ):
    TD_list = list()  # 初始化多样性列表

    time_idx = np.sort(np.unique(train_times))  # 获取排序后的唯一时间索引

    for time in tqdm(time_idx):  # 遍历每个时间点
        doc_idx = np.where(train_times == time)[0]  # 获取该时间点的文档索引
        time_vocab_idx = np.nonzero(train_bow[doc_idx].sum(0))[0]  # 获取该时间点的词汇索引
        time_vocab = np.asarray(vocab)[time_vocab_idx]  # 获取该时间点的词汇

        topics = top_words[time]  # 获取该时间点的主题
        TD_list.append(_time_slice_diversity(topics, time_vocab))  # 计算并添加时间切片多样性

    if verbose:  # 如果需要详细输出
        print(f"dynamic TD list: {TD_list}")  # 打印动态多样性列表

    return np.mean(TD_list)  # 返回平均动态多样性
