import os  # 导入操作系统模块
import json  # 导入JSON模块


def make_dir(path):  # 定义创建目录函数
    os.makedirs(path, exist_ok=True)  # 创建目录，如果已存在则不报错


def read_text(path):  # 定义读取文本文件函数
    texts = list()  # 初始化文本列表
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:  # 打开文件进行读取
        for line in file:  # 遍历文件的每一行
            texts.append(line.strip())  # 去除行首尾空白并添加到列表
    return texts  # 返回文本列表


def save_text(texts, path):  # 定义保存文本文件函数
    with open(path, 'w', encoding='utf-8') as file:  # 打开文件进行写入
        for text in texts:  # 遍历文本列表
            file.write(text.strip() + '\n')  # 写入文本并添加换行符


def read_jsonlist(path):  # 定义读取JSON列表文件函数
    data = list()  # 初始化数据列表
    with open(path, 'r', encoding='utf-8') as input_file:  # 打开输入文件
        for line in input_file:  # 遍历文件的每一行
            data.append(json.loads(line))  # 解析JSON行并添加到列表
    return data  # 返回数据列表


def save_jsonlist(list_of_json_objects, path, sort_keys=True):  # 定义保存JSON列表文件函数
    with open(path, 'w', encoding='utf-8') as output_file:  # 打开输出文件
        for obj in list_of_json_objects:  # 遍历JSON对象列表
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')  # 写入JSON对象并添加换行符


def split_text_word(texts):  # 定义文本分词函数
    texts = [text.split() for text in texts]  # 对每个文本进行分词
    return texts  # 返回分词后的文本列表
