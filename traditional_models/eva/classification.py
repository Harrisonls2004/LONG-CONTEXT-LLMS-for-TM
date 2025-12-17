import numpy as np  # 导入numpy库
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.metrics import f1_score, accuracy_score  # 导入评估指标
from collections import defaultdict  # 导入默认字典


def _cls(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale'):  # 定义分类评估函数
    if classifier == 'SVM':  # 如果使用SVM分类器
        clf = SVC(gamma=gamma)  # 创建SVM分类器
    else:  # 否则
        raise NotImplementedError  # 抛出未实现错误

    clf.fit(train_theta, train_labels)  # 训练分类器
    preds = clf.predict(test_theta)  # 预测测试集
    results = {  # 计算评估结果
        'acc': accuracy_score(test_labels, preds),  # 准确率
        'macro-F1': f1_score(test_labels, preds, average='macro')  # 宏平均F1分数
    }
    return results  # 返回评估结果


def crosslingual_cls(  # 定义跨语言分类评估函数
    train_theta_en,  # 英语训练主题分布
    train_theta_cn,  # 中文训练主题分布
    test_theta_en,  # 英语测试主题分布
    test_theta_cn,  # 中文测试主题分布
    train_labels_en,  # 英语训练标签
    train_labels_cn,  # 中文训练标签
    test_labels_en,  # 英语测试标签
    test_labels_cn,  # 中文测试标签
    classifier="SVM",  # 分类器类型
    gamma="scale"  # SVM参数
):
    intra_en = _cls(train_theta_en, test_theta_en, train_labels_en, test_labels_en, classifier, gamma)  # 英语内部分类
    intra_cn = _cls(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn, classifier, gamma)  # 中文内部分类

    cross_en = _cls(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en, classifier, gamma)  # 中文到英语跨语言分类
    cross_cn = _cls(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn, classifier, gamma)  # 英语到中文跨语言分类

    return {  # 返回跨语言分类结果
        'intra_en': intra_en,  # 英语内部分类结果
        'intra_cn': intra_cn,  # 中文内部分类结果
        'cross_en': cross_en,  # 中文到英语跨语言分类结果
        'cross_cn': cross_cn  # 英语到中文跨语言分类结果
    }


def hierarchical_cls(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale'):  # 定义层次分类评估函数
    num_layer = len(train_theta)  # 获取层数
    results = defaultdict(list)  # 初始化结果字典

    for layer in range(num_layer):  # 遍历每一层
        layer_results = _cls(train_theta[layer], test_theta[layer], train_labels, test_labels, classifier, gamma)  # 计算该层的分类结果

        for key in layer_results:  # 遍历每个评估指标
            results[key].append(layer_results[key])  # 添加到结果列表

    for key in results:  # 遍历每个评估指标
        results[key] = np.mean(results[key])  # 计算平均分数

    return results  # 返回层次分类评估结果
