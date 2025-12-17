import numpy as np  # 导入numpy库用于数值计算
from topmost.data import file_utils  # 导入文件工具模块


def get_top_words(beta, vocab, num_top_words, verbose=False):  # 定义获取主题词函数
    topic_str_list = list()  # 初始化主题词列表
    for i, topic_dist in enumerate(beta):  # 遍历每个主题分布
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]  # 获取主题词
        topic_str = ' '.join(topic_words)  # 将主题词连接成字符串
        topic_str_list.append(topic_str)  # 添加到主题词列表
        if verbose:  # 如果需要详细输出
            print('Topic {}: {}'.format(i, topic_str))  # 打印主题信息

    return topic_str_list  # 返回主题词列表


def get_stopwords_set(stopwords=[]):  # 定义获取停用词集合函数
    from topmost.data import download_dataset  # 导入数据集下载功能

    if stopwords == 'English':  # 如果是英语停用词
        from gensim.parsing.preprocessing import STOPWORDS as stopwords  # 导入gensim的英语停用词

    elif stopwords in ['mallet', 'snowball']:  # 如果是mallet或snowball停用词
        download_dataset('stopwords', cache_path='./')  # 下载停用词数据集
        path = f'./stopwords/{stopwords}_stopwords.txt'  # 构建停用词文件路径
        stopwords = file_utils.read_text(path)  # 读取停用词文件

    stopword_set = frozenset(stopwords)  # 将停用词转换为不可变集合

    return stopword_set  # 返回停用词集合


if __name__ == '__main__':  # 主程序入口
    print(list(get_stopwords_set('English'))[:10])  # 打印英语停用词前10个
    print(list(get_stopwords_set('mallet'))[:10])  # 打印mallet停用词前10个
    print(list(get_stopwords_set('snowball'))[:10])  # 打印snowball停用词前10个
    print(list(get_stopwords_set())[:10])  # 打印默认停用词前10个
