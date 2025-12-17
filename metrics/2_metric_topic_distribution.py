#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题下平均分配文本数量评估指标

评估每个主题下分配的文本数量，计算均值和方差
由于无法输出全部文档，只能输出有限数量主题下文本
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import sys

# 配置区域
INPUT_FILE = "data/NYT_sampled.csv"
JSON_FILE = "llm_analysis/results/topic_analysis_NYT_sampled_qwen_qwen3_235b_a22bfree.json"
def analyze_topic_distribution(json_file: str) -> Dict[str, Any]:
    """
    分析主题分配分布
    
    Args:
        json_file (str): 主题分析结果JSON文件路径
        
    Returns:
        Dict: 分析结果
    """
    print("📊 分析主题下平均分配文本数量...")
    print(f"   📁 JSON文件: {json_file}")
    
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取主题分配信息
        topics = data.get('topics', [])
        if not topics:
            return {"error": "JSON文件中未找到主题信息"}
        
        # 统计每个主题下的文本数量
        topic_text_counts = []
        topic_details = []
        
        for topic in topics:
            if isinstance(topic, dict):
                # 获取主题信息
                topic_num = topic.get('topic_num', 'Unknown')
                summary = topic.get('summary', 'No summary')
                
                # 统计该主题下的文本数量
                source_titles = topic.get('source_titles_with_ids', [])
                text_count = len(source_titles)
                
                topic_text_counts.append(text_count)
                topic_details.append({
                    'topic_num': topic_num,
                    'summary': summary,
                    'text_count': text_count
                })
        
        if not topic_text_counts:
            return {"error": "未找到有效的主题分配数据"}
        
        # 计算统计指标
        mean_count = np.mean(topic_text_counts)
        variance = np.var(topic_text_counts)
        std_dev = np.std(topic_text_counts)
        min_count = np.min(topic_text_counts)
        max_count = np.max(topic_text_counts)
        
        # 计算分布均匀性（变异系数）
        cv = std_dev / mean_count if mean_count > 0 else 0
        
        results = {
            "metric_name": "主题下平均分配文本数量",
            "total_topics": len(topic_text_counts),
            "mean_texts_per_topic": round(mean_count, 2),
            "variance": round(variance, 2),
            "standard_deviation": round(std_dev, 2),
            "min_texts_per_topic": int(min_count),
            "max_texts_per_topic": int(max_count),
            "coefficient_of_variation": round(cv, 4),
            "distribution_uniformity": "高" if cv < 0.3 else "中" if cv < 0.6 else "低",
            "topic_details": topic_details,
            "text_count_distribution": topic_text_counts
        }
        
        # 输出结果
        print(f"\n📊 主题分配分析结果:")
        print(f"   📈 总主题数: {results['total_topics']}")
        print(f"   📈 平均每主题文本数: {results['mean_texts_per_topic']}")
        print(f"   📈 方差: {results['variance']}")
        print(f"   📈 标准差: {results['standard_deviation']}")
        print(f"   📈 最少文本数: {results['min_texts_per_topic']}")
        print(f"   📈 最多文本数: {results['max_texts_per_topic']}")
        print(f"   📈 变异系数: {results['coefficient_of_variation']}")
        print(f"   📈 分布均匀性: {results['distribution_uniformity']}")
        
        print(f"\n📋 各主题文本分配详情 (完整列表):")
        for detail in topic_details:  # 显示所有主题，不限制数量
            print(f"   主题{detail['topic_num']}: {detail['text_count']}篇文本 - {detail['summary']}")

        # 统计文本数量分布
        from collections import Counter
        count_distribution = Counter(topic_text_counts)
        print(f"\n📊 文本数量分布统计:")
        for count, frequency in sorted(count_distribution.items()):
            print(f"   {count}篇文本: {frequency}个主题")

        # 显示详细的数量分布
        print(f"\n📈 详细数量分布:")
        print(f"   文本数量列表: {topic_text_counts}")

        # 如果有异常值，特别标出
        if len(set(topic_text_counts)) > 1:
            print(f"\n⚠️ 发现不同的文本数量:")
            unique_counts = sorted(set(topic_text_counts))
            for count in unique_counts:
                topics_with_count = [i+1 for i, c in enumerate(topic_text_counts) if c == count]
                print(f"   {count}篇文本的主题: {topics_with_count}")
        else:
            print(f"\n✅ 所有主题都有相同的文本数量: {topic_text_counts[0]}篇")
        
        return results
        
    except FileNotFoundError:
        return {"error": f"文件未找到: {json_file}"}
    except json.JSONDecodeError:
        return {"error": f"JSON文件格式错误: {json_file}"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "topic_distribution_results.json"):
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
    print("🚀 主题下平均分配文本数量评估")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON文件不存在: {JSON_FILE}")
        return
    
    # 运行分析
    results = analyze_topic_distribution(JSON_FILE)
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    # 保存结果
    save_results(results)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
