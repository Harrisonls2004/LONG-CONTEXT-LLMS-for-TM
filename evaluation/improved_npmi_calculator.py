#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPMI连贯性计算器
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ImprovedNPMICalculator:
    """改进的NPMI连贯性计算器"""
    
    def __init__(self, reference_corpus: List[str], window_size: int = 10):
        """
        初始化NPMI计算器
        
        Args:
            reference_corpus: 参考语料库
            window_size: 滑动窗口大小
        """
        self.reference_corpus = reference_corpus
        self.window_size = window_size
        self.lemmatizer = WordNetLemmatizer()
        
        # 扩展停用词列表
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update([
            'said', 'say', 'says', 'saying', 'would', 'could', 'should',
            'one', 'two', 'three', 'first', 'second', 'last', 'next',
            'new', 'old', 'good', 'bad', 'big', 'small', 'high', 'low',
            'many', 'much', 'more', 'most', 'some', 'any', 'all', 'each',
            'every', 'other', 'another', 'same', 'different', 'such',
            'way', 'ways', 'time', 'times', 'year', 'years', 'day', 'days',
            'week', 'weeks', 'month', 'months', 'hour', 'hours',
            'make', 'made', 'take', 'taken', 'get', 'got', 'give', 'given',
            'go', 'went', 'come', 'came', 'see', 'seen', 'know', 'known',
            'think', 'thought', 'find', 'found', 'use', 'used', 'work', 'worked',
            'look', 'looked', 'seem', 'seemed', 'feel', 'felt', 'become', 'became',
            'back', 'away', 'up', 'down', 'out', 'in', 'on', 'off',
            'over', 'under', 'through', 'around', 'between', 'among',
            'just', 'only', 'even', 'still', 'yet', 'already', 'now',
            'then', 'here', 'there', 'where', 'when', 'how', 'why',
            'well', 'very', 'too', 'so', 'quite', 'rather', 'really',
            'also', 'again', 'once', 'twice', 'always', 'never', 'often',
            'sometimes', 'usually', 'generally', 'particularly', 'especially'
        ])
        
        # 预处理语料库
        self.processed_corpus = self._preprocess_corpus()
        self.word_doc_freq = self._calculate_word_doc_frequencies()
        self.total_docs = len(self.reference_corpus)
        
        print(f"✓ NPMI计算器初始化完成")
        print(f"  - 语料库大小: {self.total_docs} 文档")
        print(f"  - 窗口大小: {self.window_size}")
        print(f"  - 停用词数量: {len(self.stop_words)}")
    
    def _preprocess_corpus(self) -> List[List[str]]:
        """预处理语料库：分词、词形还原、去停用词"""
        processed = []
        
        for doc in self.reference_corpus:
            # 分词
            tokens = word_tokenize(doc.lower())
            
            # 过滤和标准化
            clean_tokens = []
            for token in tokens:
                # 只保留字母词汇
                if re.match(r'^[a-zA-Z]+$', token) and len(token) > 2:
                    # 词形还原
                    lemma = self.lemmatizer.lemmatize(token)
                    # 去停用词
                    if lemma not in self.stop_words:
                        clean_tokens.append(lemma)
            
            processed.append(clean_tokens)
        
        return processed
    
    def _calculate_word_doc_frequencies(self) -> Dict[str, int]:
        """计算词汇的文档频率"""
        word_doc_freq = defaultdict(int)
        
        for doc_tokens in self.processed_corpus:
            unique_words = set(doc_tokens)
            for word in unique_words:
                word_doc_freq[word] += 1
        
        return dict(word_doc_freq)
    
    def _calculate_window_cooccurrence(self, word1: str, word2: str) -> int:
        """计算两个词在滑动窗口内的共现次数"""
        cooccur_count = 0
        
        for doc_tokens in self.processed_corpus:
            # 在每个文档中计算滑动窗口共现
            for i in range(len(doc_tokens)):
                if doc_tokens[i] == word1:
                    # 检查窗口范围内是否有word2
                    start = max(0, i - self.window_size)
                    end = min(len(doc_tokens), i + self.window_size + 1)
                    
                    window_words = doc_tokens[start:end]
                    if word2 in window_words:
                        cooccur_count += 1
                        break  # 每个文档只计算一次
        
        return cooccur_count
    
    def _calculate_npmi_pair(self, word1: str, word2: str) -> Optional[float]:
        """计算两个词的NPMI值"""
        # 标准化词汇
        word1 = self.lemmatizer.lemmatize(word1.lower())
        word2 = self.lemmatizer.lemmatize(word2.lower())
        
        # 跳过停用词
        if word1 in self.stop_words or word2 in self.stop_words:
            return None
        
        # 获取词频
        freq_w1 = self.word_doc_freq.get(word1, 0)
        freq_w2 = self.word_doc_freq.get(word2, 0)
        
        if freq_w1 == 0 or freq_w2 == 0:
            return None
        
        # 计算共现频率
        cooccur_freq = self._calculate_window_cooccurrence(word1, word2)
        
        if cooccur_freq == 0:
            return None
        
        # 计算概率
        p_w1 = freq_w1 / self.total_docs
        p_w2 = freq_w2 / self.total_docs
        p_w1_w2 = cooccur_freq / self.total_docs
        
        # 添加平滑
        epsilon = 1e-10
        p_w1_w2 = max(p_w1_w2, epsilon)
        
        # 计算PMI
        pmi = math.log(p_w1_w2 / (p_w1 * p_w2))
        
        # 计算NPMI
        npmi = pmi / (-math.log(p_w1_w2))
        
        return npmi
    
    def calculate_topic_npmi(self, topic_words: List[str], topk: int = 10) -> float:
        """计算单个主题的NPMI连贯性"""
        if len(topic_words) < 2:
            return 0.0
        
        # 只使用前topk个词
        words = topic_words[:topk]
        
        npmi_scores = []
        
        # 计算所有词对的NPMI
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                npmi = self._calculate_npmi_pair(words[i], words[j])
                if npmi is not None:
                    npmi_scores.append(npmi)
        
        return np.mean(npmi_scores) if npmi_scores else 0.0
    
    def calculate_topics_npmi(self, topics_words: List[List[str]], topk: int = 10) -> float:
        """计算多个主题的平均NPMI连贯性"""
        if not topics_words:
            return 0.0
        
        topic_scores = []
        
        for i, topic_words in enumerate(topics_words):
            score = self.calculate_topic_npmi(topic_words, topk)
            topic_scores.append(score)
            print(f"  主题 {i+1} NPMI: {score:.4f}")
        
        avg_score = np.mean(topic_scores)
        print(f"平均NPMI连贯性: {avg_score:.4f}")
        
        return avg_score
    
    def evaluate_topic_quality(self, topic_words: List[str]) -> Dict[str, any]:
        """评估主题词质量"""
        # 标准化词汇
        clean_words = []
        stop_word_count = 0
        
        for word in topic_words:
            clean_word = self.lemmatizer.lemmatize(word.lower())
            if clean_word in self.stop_words:
                stop_word_count += 1
            else:
                clean_words.append(clean_word)
        
        # 计算指标
        total_words = len(topic_words)
        clean_ratio = len(clean_words) / total_words if total_words > 0 else 0
        stop_ratio = stop_word_count / total_words if total_words > 0 else 0
        
        # 计算词汇多样性
        unique_words = len(set(clean_words))
        diversity = unique_words / len(clean_words) if clean_words else 0
        
        # 计算平均词频
        word_freqs = [self.word_doc_freq.get(word, 0) for word in clean_words]
        avg_freq = np.mean(word_freqs) if word_freqs else 0
        
        return {
            'total_words': total_words,
            'clean_words': len(clean_words),
            'stop_words': stop_word_count,
            'clean_ratio': clean_ratio,
            'stop_ratio': stop_ratio,
            'diversity': diversity,
            'avg_frequency': avg_freq,
            'quality_score': clean_ratio * diversity  # 综合质量分数
        }
    
    def analyze_topics_quality(self, topics_words: List[List[str]]) -> Dict[str, any]:
        """分析所有主题的质量"""
        quality_scores = []
        detailed_results = []
        
        for i, topic_words in enumerate(topics_words):
            quality = self.evaluate_topic_quality(topic_words)
            quality_scores.append(quality['quality_score'])
            detailed_results.append({
                'topic_id': i + 1,
                'words': topic_words[:10],  # 只显示前10个词
                **quality
            })
        
        # 统计结果
        avg_quality = np.mean(quality_scores)
        low_quality_topics = sum(1 for score in quality_scores if score < 0.5)
        
        return {
            'average_quality': avg_quality,
            'low_quality_count': low_quality_topics,
            'total_topics': len(topics_words),
            'detailed_results': detailed_results
        }

def compare_npmi_methods(topics_words: List[List[str]], reference_corpus: List[str]) -> Dict[str, float]:
    """比较不同NPMI计算方法的结果"""
    print("=== NPMI计算方法比较 ===")
    
    # 方法1：改进的NPMI计算器
    improved_calculator = ImprovedNPMICalculator(reference_corpus)
    improved_score = improved_calculator.calculate_topics_npmi(topics_words)
    
    # 方法2：原始简单计算（用于对比）
    from topic_evaluation_Tra import TraTopicEvaluator
    original_evaluator = TraTopicEvaluator(reference_corpus)
    original_score = original_evaluator.calculate_npmi_coherence_window(topics_words)
    
    results = {
        'improved_npmi': improved_score,
        'original_npmi': original_score,
        'improvement': improved_score - original_score
    }
    
    print(f"\n📊 NPMI计算结果比较:")
    print(f"  改进方法: {improved_score:.4f}")
    print(f"  原始方法: {original_score:.4f}")
    print(f"  改进幅度: {results['improvement']:+.4f}")
    
    return results

if __name__ == "__main__":
    # 测试示例
    test_topics = [
        ["president", "election", "vote", "campaign", "political"],
        ["health", "medical", "hospital", "doctor", "patient"],
        ["economy", "market", "financial", "business", "trade"]
    ]
    
    test_corpus = [
        "The president won the election with a strong campaign.",
        "Medical professionals work in hospitals to help patients.",
        "The economy depends on financial markets and business trade."
    ]
    
    calculator = ImprovedNPMICalculator(test_corpus)
    score = calculator.calculate_topics_npmi(test_topics)
    print(f"测试NPMI分数: {score:.4f}")
