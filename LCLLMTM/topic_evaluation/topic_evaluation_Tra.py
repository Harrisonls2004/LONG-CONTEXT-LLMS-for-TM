#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM主题分析结果评测模块
合并RLForTopic和TopMost的评测指标，适配LLM输出结果
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class TraTopicEvaluator:
    """LLM主题分析结果评测器"""
    
    def __init__(self, reference_corpus: List[str] = None):
        """
        初始化评测器
        
        Args:
            reference_corpus: 参考语料库（用于计算NPMI等指标）
        """
        self.reference_corpus = reference_corpus
        self.word_counts = None
        self.doc_counts = None
        self.total_docs = 0
        
        if reference_corpus:
            self._build_corpus_stats()
    
    def _build_corpus_stats(self):
        """构建语料库统计信息"""
        self.word_counts = Counter()
        self.doc_counts = Counter()
        self.total_docs = len(self.reference_corpus)
        
        for doc in self.reference_corpus:
            words = doc.lower().split()
            doc_words = set(words)
            
            for word in words:
                self.word_counts[word] += 1
            
            for word in doc_words:
                self.doc_counts[word] += 1
    
    def evaluate_all(self, topics: List[Dict], topk: int = 8) -> Dict[str, float]:
        """
        综合评测所有指标 - 只使用确实可用的评测指标

        Args:
            topics: LLM输出的主题列表，每个主题包含keywords字段
            topk: 评测时使用的top-k关键词数量

        Returns:
            包含所有评测指标的字典
        """
        results = {}

        # 提取主题词列表
        topic_words = []
        for topic in topics:
            keywords = topic.get('keywords', [])[:topk]
            topic_words.append(keywords)

        # ========== 4种可用评测指标 ==========
        # 1. 主题多样性 (Topic Diversity) - TopMost ✅
        results['topic_diversity'] = self.calculate_topic_diversity(topic_words)

        # # 2. RBO相似度 (Rank-Biased Overlap) - RL-for-topic ✅
        # if len(topic_words) >= 4:  # 至少需要4个主题才能分成两组
        #     # 将主题列表分成两半进行RBO比较
        #     mid = len(topic_words) // 2
        #     topic_words_1 = topic_words[:mid]
        #     topic_words_2 = topic_words[mid:mid*2]  # 确保两个列表长度相同
        #     if len(topic_words_1) == len(topic_words_2) and len(topic_words_1) > 0:
        #         results['rbo_similarity'] = self.calculate_rbo(topic_words_1, topic_words_2)
        # else:
        #     results['rbo_similarity'] = None  # 主题数量不足

        # 3. Word2Vec连贯性 - RL-for-topic ✅ (可选)
        # try:
            # results['word2vec_coherence'] = self.calculate_word2vec_coherence(topic_words)
        # except:
        #     results['word2vec_coherence'] = None

        # 4. NPMI连贯性 (window=10) - RL-for-topic ✅
        if self.reference_corpus:
            results['npmi_coherence_window'] = self.calculate_npmi_coherence_window(topic_words, window_size=10)


        return results
    
    def calculate_topic_diversity(self, topic_words: List[List[str]]) -> float:
        """
        计算主题多样性 (Topic Diversity) - TopMost标准实现
        TD = |unique_words| / |total_words|

        Args:
            topic_words: 每个主题的关键词列表

        Returns:
            多样性分数 (0-1，越高越好)
        """
        if not topic_words:
            return 0.0

        all_words = set()
        total_words = 0

        for words in topic_words:
            all_words.update(words)
            total_words += len(words)

        if total_words == 0:
            return 0.0

        return len(all_words) / total_words

    def calculate_word_uniqueness(self, topic_words: List[List[str]]) -> float:
        """
        计算词汇独特性 - TopMost指标
        衡量只出现在一个主题中的词汇比例
        """
        if not topic_words:
            return 0.0

        word_freq = Counter()
        for words in topic_words:
            for word in words:
                word_freq[word] += 1

        unique_words = sum(1 for count in word_freq.values() if count == 1)
        total_unique_words = len(word_freq)

        return unique_words / total_unique_words if total_unique_words > 0 else 0.0
    
    def calculate_unique_words_ratio(self, topic_words: List[List[str]]) -> float:
        """计算独特词汇比例"""
        if not topic_words:
            return 0.0
        
        word_freq = Counter()
        for words in topic_words:
            for word in words:
                word_freq[word] += 1
        
        unique_words = sum(1 for count in word_freq.values() if count == 1)
        total_words = len(word_freq)
        
        return unique_words / total_words if total_words > 0 else 0.0

    def calculate_word2vec_coherence(self, topic_words: List[List[str]]) -> float:
        """
        计算Word2Vec连贯性 - RL-for-topic-models标准指标
        使用词向量余弦相似度衡量主题内词的语义一致性
        """
        try:
            # 尝试导入gensim和下载预训练模型
            import gensim.downloader as api

            # 使用预训练的Word2Vec模型
            model = api.load("word2vec-google-news-300")

            coherence_scores = []

            for words in topic_words:
                if len(words) < 2:
                    continue

                # 过滤模型中存在的词
                valid_words = [word for word in words if word in model.key_to_index]

                if len(valid_words) < 2:
                    continue

                # 计算词对的余弦相似度
                similarities = []
                for i in range(len(valid_words)):
                    for j in range(i + 1, len(valid_words)):
                        try:
                            sim = model.similarity(valid_words[i], valid_words[j])
                            similarities.append(sim)
                        except:
                            continue

                if similarities:
                    coherence_scores.append(np.mean(similarities))

            return np.mean(coherence_scores) if coherence_scores else 0.0

        except Exception as e:
            # 如果无法加载Word2Vec模型，返回None
            return None



    def calculate_npmi_coherence_window(self, topic_words: List[List[str]], window_size: int = 10) -> float:
        """
        计算NPMI连贯性 (window=10) - RL-for-topic-models标准指标
        基于滑动窗口的词汇共现统计
        """
        if not self.reference_corpus or not topic_words:
            return 0.0

        # 简化的实现：基于文档级别的共现
        coherence_scores = []

        for words in topic_words:
            if len(words) < 2:
                continue

            # 计算词对的NPMI
            npmi_scores = []
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i].lower(), words[j].lower()

                    # 计算词汇在文档中的出现次数
                    count_w1 = sum(1 for doc in self.reference_corpus if w1 in doc.lower())
                    count_w2 = sum(1 for doc in self.reference_corpus if w2 in doc.lower())
                    count_both = sum(1 for doc in self.reference_corpus if w1 in doc.lower() and w2 in doc.lower())

                    if count_w1 > 0 and count_w2 > 0 and count_both > 0:
                        total_docs = len(self.reference_corpus)
                        p_w1 = count_w1 / total_docs
                        p_w2 = count_w2 / total_docs
                        p_both = count_both / total_docs

                        # 计算PMI
                        pmi = math.log(p_both / (p_w1 * p_w2))
                        # 标准化为NPMI
                        npmi = pmi / (-math.log(p_both))
                        npmi_scores.append(npmi)

            if npmi_scores:
                coherence_scores.append(np.mean(npmi_scores))

        return np.mean(coherence_scores) if coherence_scores else 0.0


        """计算符合要求的主题比例（5-8个关键词）"""
        if not topics:
            return 0.0
        
        valid_count = sum(1 for topic in topics 
                         if 5 <= len(topic.get('keywords', [])) <= 8)
        
        return valid_count / len(topics)
    
    
    def _calculate_npmi(self, word1: str, word2: str) -> Optional[float]:
        """计算两个词的NPMI值"""
        if not self.doc_counts:
            return None
        
        # 获取词频
        count_w1 = self.doc_counts.get(word1, 0)
        count_w2 = self.doc_counts.get(word2, 0)
        
        if count_w1 == 0 or count_w2 == 0:
            return None
        
        # 计算共现频率
        cooccur_count = 0
        for doc in self.reference_corpus:
            words = set(doc.lower().split())
            if word1 in words and word2 in words:
                cooccur_count += 1
        
        if cooccur_count == 0:
            return None
        
        # 计算概率
        p_w1 = count_w1 / self.total_docs
        p_w2 = count_w2 / self.total_docs
        p_w1_w2 = cooccur_count / self.total_docs
        
        # 计算PMI
        pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
        
        # 计算NPMI
        npmi = pmi / (-math.log(p_w1_w2))
        
        return npmi
    
    def calculate_pmi_coherence(self, topic_words: List[List[str]]) -> float:
        """计算PMI连贯性"""
        if not self.reference_corpus or not topic_words:
            return 0.0
        
        coherence_scores = []
        
        for words in topic_words:
            if len(words) < 2:
                continue
            
            word_pairs = list(combinations(words, 2))
            pair_scores = []
            
            for w1, w2 in word_pairs:
                pmi = self._calculate_pmi(w1, w2)
                if pmi is not None:
                    pair_scores.append(pmi)
            
            if pair_scores:
                coherence_scores.append(np.mean(pair_scores))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_pmi(self, word1: str, word2: str) -> Optional[float]:
        """计算两个词的PMI值"""
        if not self.doc_counts:
            return None
        
        count_w1 = self.doc_counts.get(word1, 0)
        count_w2 = self.doc_counts.get(word2, 0)
        
        if count_w1 == 0 or count_w2 == 0:
            return None
        
        cooccur_count = 0
        for doc in self.reference_corpus:
            words = set(doc.lower().split())
            if word1 in words and word2 in words:
                cooccur_count += 1
        
        if cooccur_count == 0:
            return None
        
        p_w1 = count_w1 / self.total_docs
        p_w2 = count_w2 / self.total_docs
        p_w1_w2 = cooccur_count / self.total_docs
        
        return math.log(p_w1_w2 / (p_w1 * p_w2))
    

    def _calculate_cv_coherence(self, word1: str, word2: str) -> Optional[float]:
        """计算C_V连贯性分数"""
        if not self.doc_counts:
            return None

        # 简化的C_V实现，使用文档共现
        count_w1 = self.doc_counts.get(word1.lower(), 0)
        count_w2 = self.doc_counts.get(word2.lower(), 0)

        if count_w1 == 0 or count_w2 == 0:
            return None

        # 计算共现
        cooccur_count = 0
        for doc in self.reference_corpus:
            words = set(doc.lower().split())
            if word1.lower() in words and word2.lower() in words:
                cooccur_count += 1

        if cooccur_count == 0:
            return None

        # 简化的C_V计算
        p_w1_w2 = cooccur_count / self.total_docs
        p_w1 = count_w1 / self.total_docs
        p_w2 = count_w2 / self.total_docs

        if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
            return math.log((p_w1_w2 + 1e-10) / (p_w1 * p_w2 + 1e-10))

        return None


    def calculate_word2vec_coherence(self, topic_words: List[List[str]]) -> float:
        """
        计算Word2Vec连贯性 - RL-for-topic-models指标
        使用词向量余弦相似度衡量主题内词的语义一致性
        """
        try:
            # 尝试导入gensim和下载预训练模型
            import gensim.downloader as api

            # 使用预训练的Word2Vec模型
            model = api.load("word2vec-google-news-300")

            coherence_scores = []

            for words in topic_words:
                if len(words) < 2:
                    continue

                # 过滤模型中存在的词
                valid_words = [word for word in words if word in model.key_to_index]

                if len(valid_words) < 2:
                    continue

                # 计算词对的余弦相似度
                similarities = []
                for i in range(len(valid_words)):
                    for j in range(i + 1, len(valid_words)):
                        try:
                            sim = model.similarity(valid_words[i], valid_words[j])
                            similarities.append(sim)
                        except:
                            continue

                if similarities:
                    coherence_scores.append(np.mean(similarities))

            return np.mean(coherence_scores) if coherence_scores else 0.0

        except Exception as e:
            # 如果无法加载Word2Vec模型，返回None
            return None

    def _calculate_cv_score(self, word1: str, word2: str) -> Optional[float]:
        """计算C_V连贯性分数"""
        if not self.doc_counts:
            return None

        # 简化的C_V实现，使用文档共现
        count_w1 = self.doc_counts.get(word1, 0)
        count_w2 = self.doc_counts.get(word2, 0)

        if count_w1 == 0 or count_w2 == 0:
            return None

        # 计算共现
        cooccur_count = 0
        for doc in self.reference_corpus:
            words = set(doc.lower().split())
            if word1 in words and word2 in words:
                cooccur_count += 1

        if cooccur_count == 0:
            return None

        # 简化的C_V计算
        p_w1_w2 = cooccur_count / self.total_docs
        p_w1 = count_w1 / self.total_docs
        p_w2 = count_w2 / self.total_docs

        if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
            return math.log((p_w1_w2 + 1e-10) / (p_w1 * p_w2 + 1e-10))

        return None

    def _calculate_umass_score(self, word1: str, word2: str) -> Optional[float]:
        """计算UMass连贯性分数"""
        if not self.doc_counts:
            return None

        count_w1 = self.doc_counts.get(word1, 0)
        count_w2 = self.doc_counts.get(word2, 0)

        if count_w1 == 0 or count_w2 == 0:
            return None

        # 计算共现
        cooccur_count = 0
        for doc in self.reference_corpus:
            words = set(doc.lower().split())
            if word1 in words and word2 in words:
                cooccur_count += 1

        # UMass公式: log((D(w_i, w_j) + 1) / D(w_j))
        return math.log((cooccur_count + 1) / count_w2)
    
    def calculate_max_topic_overlap(self, topic_words: List[List[str]]) -> float:
        """向后兼容的方法名"""
        return self.calculate_max_jaccard_overlap(topic_words)

    def calculate_topic_similarity(self, topic_words: List[List[str]]) -> float:
        """向后兼容的方法名"""
        return self.calculate_jaccard_similarity(topic_words)

    def calculate_max_jaccard_overlap_old(self, topic_words: List[List[str]]) -> float:
        """计算最大主题重叠度"""
        if len(topic_words) < 2:
            return 0.0
        
        max_overlap = 0.0
        
        for i in range(len(topic_words)):
            for j in range(i + 1, len(topic_words)):
                set1 = set(topic_words[i])
                set2 = set(topic_words[j])
                
                if len(set1) > 0 and len(set2) > 0:
                    overlap = len(set1.intersection(set2)) / min(len(set1), len(set2))
                    max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def calculate_rbo(self, list1: List[List[str]], list2: List[List[str]], p: float = 0.9) -> float:
        """
        计算RBO (Rank-Biased Overlap)
        用于比较两个模型的主题词排序相似度
        
        Args:
            list1, list2: 两个主题词列表
            p: RBO参数，控制对排序位置的重视程度

        Returns:
            RBO分数 (0-1)
        """
        if len(list1) != len(list2):
            return 0.0

        rbo_scores = []
        
        for i in range(min(len(list1), len(list2))):
            words1 = list1[i]
            words2 = list2[i]
            
            rbo = self._rbo_score(words1, words2, p)
            rbo_scores.append(rbo)
        
        return np.mean(rbo_scores) if rbo_scores else 0.0
    
    def _rbo_score(self, list1: List[str], list2: List[str], p: float) -> float:
        """计算两个列表的RBO分数"""
        if not list1 or not list2:
            return 0.0
        
        max_len = max(len(list1), len(list2))
        min_len = min(len(list1), len(list2))
        
        # 计算重叠
        overlap = 0.0
        for d in range(1, min_len + 1):
            set1 = set(list1[:d])
            set2 = set(list2[:d])
            overlap += len(set1.intersection(set2)) / d * (p ** (d - 1))
        
        # 添加尾部权重
        if max_len > min_len:
            overlap += (len(set(list1[:min_len]).intersection(set(list2[:min_len]))) / min_len) * (p ** min_len) * (1 - p) / (1 - p)
        
        return (1 - p) * overlap
    
    def print_evaluation_report(self, results: Dict[str, float]):
        """打印评测报告"""
        print("\n" + "="*80)
        print("📊 LLM主题分析评测报告")
        print("="*80)
        
        # print(f"\n🎯 主题质量指标:")
        # print(f"   平均关键词数量: {results.get('avg_keywords_per_topic', 0):.2f}")
        # print(f"   关键词数量方差: {results.get('keyword_count_variance', 0):.2f}")
        # print(f"   符合要求主题比例: {results.get('valid_topics_ratio', 0):.2%}")
        
        print(f"\n🎨 主题多样性指标:")
        print(f"   主题多样性: {results.get('diversity', 0):.3f}")
        print(f"   独特词汇比例: {results.get('unique_words_ratio', 0):.3f}")
        print(f"   平均主题相似度: {results.get('avg_topic_similarity', 0):.3f}")
        print(f"   最大主题重叠度: {results.get('max_topic_overlap', 0):.3f}")
        
        if 'npmi_coherence' in results:
            print(f"\n🔗 语义连贯性指标:")
            print(f"   NPMI连贯性: {results.get('npmi_coherence', 0):.3f}")
            print(f"   PMI连贯性: {results.get('pmi_coherence', 0):.3f}")
        
        print(f"\n📈 评测总结:")
        quality_score = (results.get('valid_topics_ratio', 0) + 
                        results.get('diversity', 0) + 
                        (1 - results.get('avg_topic_similarity', 1))) / 3
        print(f"   综合质量分数: {quality_score:.3f}")
        
        # # 给出建议
        # print(f"\n💡 改进建议:")
        # if results.get('valid_topics_ratio', 0) < 0.8:
        #     print("   - 关键词数量控制需要改进，建议优化提示词")
        # if results.get('diversity', 0) < 0.7:
        #     print("   - 主题多样性较低，建议增加主题数量或改进算法")
        # if results.get('avg_topic_similarity', 0) > 0.3:
        #     print("   - 主题间相似度较高，建议提高主题区分度")
        # if results.get('npmi_coherence', 0) < 0.1:
        #     print("   - 语义连贯性较低，建议改进主题词选择策略")
