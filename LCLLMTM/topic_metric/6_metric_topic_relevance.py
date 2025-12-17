#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题文本相关性/幻觉率评估指标

这个指标直接把主题的关键词keywords那一列和文本的关键词比对
如果出现重复的那就是相关的，用于检测主题生成的幻觉率
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set
import os
import re
from collections import Counter

# 配置区域
CSV_FILE = "C:/Users/1/Desktop/TopMost/LCLLMTM/data4LCLLM/NYT_sampled.csv"

def extract_keywords_from_text(text: str) -> Set[str]:
    """
    从文本中提取关键词
    
    Args:
        text (str): 输入文本
        
    Returns:
        Set[str]: 关键词集合
    """
    # 简单的关键词提取：去除标点符号，转换为小写，分割单词
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # 过滤常见停用词
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use'}
    keywords = set(word for word in words if word not in stop_words and len(word) > 3)
    return keywords

def analyze_topic_relevance(csv_file: str) -> Dict[str, Any]:
    """
    分析主题文本相关性/幻觉率
    
    Args:
        csv_file (str): 包含topic和keywords列的CSV文件路径
        
    Returns:
        Dict: 分析结果
    """
    print("🔍 分析主题文本相关性/幻觉率...")
    print(f"   📁 CSV文件: {csv_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查必需的列
        required_columns = ['text', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"CSV文件缺少必需的列: {missing_columns}"}
        
        # 如果有topic列，也使用它
        has_topic_column = 'topic' in df.columns
        
        relevance_scores = []
        hallucination_cases = []
        detailed_analysis = []
        
        print(f"   📊 分析 {len(df)} 行数据...")
        
        for idx, row in df.iterrows():
            try:
                # 获取文本和关键词
                text = str(row['text']) if pd.notna(row['text']) else ""
                keywords_str = str(row['keywords']) if pd.notna(row['keywords']) else ""
                topic = str(row['topic']) if has_topic_column and pd.notna(row['topic']) else ""
                
                # 解析关键词（假设是JSON格式的列表）
                try:
                    if keywords_str.startswith('[') and keywords_str.endswith(']'):
                        # JSON格式
                        keywords_list = json.loads(keywords_str)
                    else:
                        # 逗号分隔格式
                        keywords_list = [kw.strip().strip("'\"") for kw in keywords_str.split(',')]
                    
                    # 清理和标准化关键词
                    topic_keywords = set()
                    for kw in keywords_list:
                        if isinstance(kw, str) and kw.strip():
                            topic_keywords.add(kw.strip().lower())
                
                except (json.JSONDecodeError, ValueError):
                    # 如果解析失败，尝试简单分割
                    keywords_list = keywords_str.split(',')
                    topic_keywords = set(kw.strip().lower() for kw in keywords_list if kw.strip())
                
                # 从文本中提取关键词
                text_keywords = extract_keywords_from_text(text)
                
                # 添加主题词到关键词集合
                if topic:
                    topic_words = extract_keywords_from_text(topic)
                    topic_keywords.update(topic_words)
                
                # 计算重叠关键词
                overlap_keywords = topic_keywords.intersection(text_keywords)
                
                # 计算相关性得分
                if len(topic_keywords) > 0:
                    relevance_score = len(overlap_keywords) / len(topic_keywords)
                else:
                    relevance_score = 0
                
                # 计算幻觉关键词（主题中有但文本中没有的关键词）
                hallucination_keywords = topic_keywords - text_keywords
                hallucination_ratio = len(hallucination_keywords) / len(topic_keywords) if len(topic_keywords) > 0 else 0
                
                relevance_scores.append(relevance_score)
                
                # 记录详细分析
                analysis_detail = {
                    "row_index": idx,
                    "topic_keywords_count": len(topic_keywords),
                    "text_keywords_count": len(text_keywords),
                    "overlap_count": len(overlap_keywords),
                    "relevance_score": round(relevance_score, 4),
                    "hallucination_ratio": round(hallucination_ratio, 4),
                    "overlap_keywords": list(overlap_keywords),
                    "hallucination_keywords": list(hallucination_keywords)
                }
                detailed_analysis.append(analysis_detail)
                
                # 记录严重幻觉案例
                if hallucination_ratio > 0.5:  # 超过50%的关键词是幻觉
                    hallucination_cases.append({
                        "row_index": idx,
                        "hallucination_ratio": hallucination_ratio,
                        "hallucination_keywords": list(hallucination_keywords),
                        "topic": topic[:100] if topic else "N/A",
                        "text_preview": text[:200] if text else "N/A"
                    })
                
            except Exception as e:
                print(f"   ⚠️ 处理第 {idx} 行时出错: {str(e)}")
                continue
        
        if not relevance_scores:
            return {"error": "没有成功分析的数据行"}
        
        # 计算整体统计
        mean_relevance = np.mean(relevance_scores)
        median_relevance = np.median(relevance_scores)
        std_relevance = np.std(relevance_scores)
        
        # 计算幻觉率统计
        hallucination_ratios = [detail['hallucination_ratio'] for detail in detailed_analysis]
        mean_hallucination = np.mean(hallucination_ratios)
        
        # 分类评估
        high_relevance_count = sum(1 for score in relevance_scores if score >= 0.7)
        medium_relevance_count = sum(1 for score in relevance_scores if 0.3 <= score < 0.7)
        low_relevance_count = sum(1 for score in relevance_scores if score < 0.3)
        
        results = {
            "metric_name": "主题文本相关性/幻觉率",
            "total_analyzed": len(relevance_scores),
            "mean_relevance_score": round(mean_relevance, 4),
            "median_relevance_score": round(median_relevance, 4),
            "std_relevance_score": round(std_relevance, 4),
            "mean_hallucination_ratio": round(mean_hallucination, 4),
            "relevance_distribution": {
                "high_relevance": high_relevance_count,
                "medium_relevance": medium_relevance_count,
                "low_relevance": low_relevance_count,
                "high_relevance_percentage": round(high_relevance_count / len(relevance_scores) * 100, 2),
                "low_relevance_percentage": round(low_relevance_count / len(relevance_scores) * 100, 2)
            },
            "hallucination_analysis": {
                "severe_hallucination_cases": len(hallucination_cases),
                "severe_hallucination_percentage": round(len(hallucination_cases) / len(relevance_scores) * 100, 2)
            },
            "evaluation": {
                "relevance_quality": "优秀" if mean_relevance >= 0.7 else "良好" if mean_relevance >= 0.5 else "一般" if mean_relevance >= 0.3 else "较差",
                "hallucination_severity": "轻微" if mean_hallucination <= 0.2 else "中等" if mean_hallucination <= 0.4 else "严重",
                "overall_assessment": "优秀" if mean_relevance >= 0.6 and mean_hallucination <= 0.3 else "良好" if mean_relevance >= 0.4 and mean_hallucination <= 0.5 else "需改进"
            },
            "sample_hallucination_cases": hallucination_cases[:5],  # 显示前5个严重案例
            "detailed_statistics": {
                "min_relevance": round(min(relevance_scores), 4),
                "max_relevance": round(max(relevance_scores), 4),
                "relevance_scores_sample": relevance_scores[:10]  # 前10个得分样例
            }
        }
        
        # 输出结果
        print(f"\n🔍 主题文本相关性/幻觉率分析结果:")
        print(f"   📊 分析样本数: {results['total_analyzed']}")
        print(f"   📊 平均相关性得分: {results['mean_relevance_score']:.4f}")
        print(f"   📊 平均幻觉率: {results['mean_hallucination_ratio']:.4f}")
        print(f"   📊 高相关性比例: {results['relevance_distribution']['high_relevance_percentage']:.1f}%")
        print(f"   📊 低相关性比例: {results['relevance_distribution']['low_relevance_percentage']:.1f}%")
        print(f"   📊 严重幻觉案例: {results['hallucination_analysis']['severe_hallucination_cases']} 个")
        print(f"   📊 相关性质量: {results['evaluation']['relevance_quality']}")
        print(f"   📊 幻觉严重程度: {results['evaluation']['hallucination_severity']}")
        print(f"   📊 整体评估: {results['evaluation']['overall_assessment']}")
        
        return results
        
    except FileNotFoundError:
        return {"error": f"文件未找到: {csv_file}"}
    except pd.errors.EmptyDataError:
        return {"error": "CSV文件为空"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "topic_relevance_results.json"):
    """保存结果到文件"""
    try:
        # 确保输出目录存在
        output_dir = "C:/Users/1/Desktop/TopMost/metric_result"
        os.makedirs(output_dir, exist_ok=True)

        # 构建完整路径
        full_path = os.path.join(output_dir, output_file)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {full_path}")
    except Exception as e:
        print(f"❌ 保存结果失败: {str(e)}")

def main():
    """主函数"""
    print("🚀 主题文本相关性/幻觉率评估")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV文件不存在: {CSV_FILE}")
        return
    
    # 运行分析
    results = analyze_topic_relevance(CSV_FILE)
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    # 保存结果
    save_results(results)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
