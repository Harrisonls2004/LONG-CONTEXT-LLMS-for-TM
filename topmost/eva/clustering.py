import numpy as np  # 导入numpy库
from collections import defaultdict  # 导入默认字典
from sklearn import metrics  # 导入sklearn评估指标


def purity_score(y_true, y_pred):  # 定义纯度分数函数
    # compute contingency matrix (also called confusion matrix)  # 计算列联矩阵（也称为混淆矩阵）
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)  # 计算列联矩阵
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)  # 计算纯度分数


def clustering_metrics(labels, preds):  # 定义聚类评估指标函数
    metrics_func = [  # 定义评估指标列表
        {
            'name': 'Purity',  # 纯度
            'method': purity_score  # 纯度计算方法
        },
        {
            'name': 'NMI',  # 归一化互信息
            'method': metrics.cluster.normalized_mutual_info_score  # NMI计算方法
        },
    ]

    results = dict()  # 初始化结果字典
    for func in metrics_func:  # 遍历每个评估指标
        results[func['name']] = func['method'](labels, preds)  # 计算评估指标

    return results  # 返回评估结果


def _clustering(theta, labels):  # 定义聚类评估函数
    preds = np.argmax(theta, axis=1)  # 获取预测标签
    return clustering_metrics(labels, preds)  # 返回聚类评估指标


def hierarchical_clustering(test_theta, test_labels):  # 定义层次聚类评估函数
    num_layer = len(test_theta)  # 获取层数
    results = defaultdict(list)  # 初始化结果字典

    for layer in range(num_layer):  # 遍历每一层
        layer_results = _clustering(test_theta[layer], test_labels)  # 计算该层的聚类结果

        for key in layer_results:  # 遍历每个评估指标
            results[key].append(layer_results[key])  # 添加到结果列表

    for key in results:  # 遍历每个评估指标
        results[key] = np.mean(results[key])  # 计算平均分数

    return results  # 返回层次聚类评估结果
