#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高频主题优先生成评估指标

判断生成的主题是否优先为源文本中出现最频繁的主题
统计每个主题下面的文本数量均值和方差
均值越大，证明生成了高频主题；方差越小，证明效果稳定
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from collections import Counter
import os

# 配置区域
INPUT_FILE = "data/NYT_sampled.csv"
JSON_FILE = "llm_analysis/results/topic_analysis_NYT_sampled_meta_llama_llama_4_maverick.json"
GROUND_TRUTH_FILE = ""  # 如果有真实标注文件

def analyze_high_frequency_priority(json_file: str, csv_file: str = None) -> Dict[str, Any]:
    """
    评估高频主题优先生成
    
    Args:
        json_file (str): 生成的主题分析结果JSON文件
        csv_file (str): 原始CSV文件路径（可选，用于对比）
        
    Returns:
        Dict: 评估结果
    """
    print("🔥 评估高频主题优先生成...")
    print(f"   📁 JSON文件: {json_file}")
    if csv_file:
        print(f"   📁 CSV文件: {csv_file}")
    
    try:
        # 读取生成的主题分析结果
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        topics = data.get('topics', [])
        if not topics:
            return {"error": "JSON文件中未找到主题信息"}
        
        # 统计每个主题的文本分配数量
        topic_frequencies = []
        topic_info = []
        
        for topic in topics:
            if isinstance(topic, dict):
                topic_num = topic.get('topic_num', 'Unknown')
                summary = topic.get('summary', 'No summary')
                keywords = topic.get('keywords', [])
                source_titles = topic.get('source_titles_with_ids', [])
                
                frequency = len(source_titles)
                topic_frequencies.append(frequency)
                
                topic_info.append({
                    'topic_num': topic_num,
                    'summary': summary,
                    'keywords': keywords,
                    'frequency': frequency,
                    'source_count': len(source_titles)
                })
        
        if not topic_frequencies:
            return {"error": "未找到有效的主题频率数据"}
        
        # 计算统计指标
        mean_frequency = np.mean(topic_frequencies)
        variance = np.var(topic_frequencies)
        std_dev = np.std(topic_frequencies)
        
        # 排序主题按频率
        sorted_topics = sorted(topic_info, key=lambda x: x['frequency'], reverse=True)
        
        # 计算高频主题比例（前20%的主题）
        top_20_percent_count = max(1, len(sorted_topics) // 5)
        high_freq_topics = sorted_topics[:top_20_percent_count]
        high_freq_total = sum(t['frequency'] for t in high_freq_topics)
        total_frequency = sum(topic_frequencies)
        high_freq_ratio = high_freq_total / total_frequency if total_frequency > 0 else 0
        
        # 评估稳定性（方差越小越稳定）
        stability_score = 1 / (1 + variance/mean_frequency) if mean_frequency > 0 else 0
        
        # 评估高频优先性（均值越大越好）
        priority_score = mean_frequency / max(topic_frequencies) if topic_frequencies else 0
        
        results = {
            "metric_name": "高频主题优先生成",
            "total_topics": len(topic_frequencies),
            "mean_frequency": round(mean_frequency, 2),
            "variance": round(variance, 2),
            "standard_deviation": round(std_dev, 2),
            "high_frequency_ratio": round(high_freq_ratio, 4),
            "stability_score": round(stability_score, 4),
            "priority_score": round(priority_score, 4),
            "frequency_distribution": topic_frequencies,
            "top_topics": high_freq_topics[:10],  # 前10个高频主题
            "evaluation": {
                "high_frequency_priority": "优秀" if priority_score > 0.7 else "良好" if priority_score > 0.5 else "一般",
                "stability": "稳定" if stability_score > 0.7 else "中等" if stability_score > 0.5 else "不稳定",
                "overall_assessment": "优秀" if (priority_score > 0.7 and stability_score > 0.7) else "良好" if (priority_score > 0.5 and stability_score > 0.5) else "需改进"
            }
        }
        
        # 输出结果
        print(f"\n🔥 高频主题优先生成分析结果:")
        print(f"   📊 总主题数: {results['total_topics']}")
        print(f"   📊 平均频率: {results['mean_frequency']}")
        print(f"   📊 方差: {results['variance']} (越小越稳定)")
        print(f"   📊 标准差: {results['standard_deviation']}")
        print(f"   📊 高频主题占比: {results['high_frequency_ratio']:.2%}")
        print(f"   📊 稳定性得分: {results['stability_score']:.4f}")
        print(f"   📊 优先性得分: {results['priority_score']:.4f}")
        print(f"   📊 整体评估: {results['evaluation']['overall_assessment']}")
        
        print(f"\n🏆 前10个高频主题:")
        for i, topic in enumerate(high_freq_topics[:10], 1):
            print(f"   {i}. 主题{topic['topic_num']}: {topic['frequency']}篇 - {topic['summary'][:50]}...")
        
        return results
        
    except FileNotFoundError:
        return {"error": f"文件未找到: {json_file}"}
    except json.JSONDecodeError:
        return {"error": f"JSON文件格式错误: {json_file}"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "high_frequency_priority_results.json"):
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
    print("🚀 高频主题优先生成评估")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON文件不存在: {JSON_FILE}")
        return
    
    # 运行分析
    results = analyze_high_frequency_priority(JSON_FILE, INPUT_FILE if os.path.exists(INPUT_FILE) else None)
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    # 保存结果
    save_results(results)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
