#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入文本位置偏好分析评估指标

分别计算前30%、中40%、后30%输入文本出现在答案中的比例
评估模型是否存在位置偏好现象（如前端偏好、中间忽视等）
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import re

# 配置区域
INPUT_FILE = "data/NYT_sampled.csv"
JSON_FILE = "llm_analysis/results/topic_analysis_NYT_sampled_qwen_qwen3_235b_a22bfree.json"

def analyze_position_bias_ratio(csv_file: str, json_file: str) -> Dict[str, Any]:
    """
    分析输入文本位置偏好比例（前30% | 中40% | 后30%）

    Args:
        csv_file (str): 输入CSV文件路径
        json_file (str): 生成结果JSON文件路径

    Returns:
        Dict: 分析结果
    """
    print("🔍 分析输入文本位置偏好比例（前30% | 中40% | 后30%）...")
    print(f"   📁 CSV文件: {csv_file}")
    print(f"   📁 JSON文件: {json_file}")
    
    try:
        # 读取输入CSV文件
        df = pd.read_csv(csv_file)
        if 'text' not in df.columns or 'id' not in df.columns:
            return {"error": "CSV文件必须包含'text'和'id'列"}
        
        # 读取生成结果JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 获取输入文本总数
        total_texts = len(df)

        # 计算三个区间的文本数量和ID集合
        front_30_count = int(total_texts * 0.3)
        middle_40_start = front_30_count
        middle_40_end = int(total_texts * 0.7)
        back_30_start = middle_40_end

        # 提取各区间的ID集合
        front_30_ids = set(df.iloc[:front_30_count]['id'].astype(str))
        middle_40_ids = set(df.iloc[middle_40_start:middle_40_end]['id'].astype(str))
        back_30_ids = set(df.iloc[back_30_start:]['id'].astype(str))

        print(f"   📊 总文本数: {total_texts}")
        print(f"   📊 前30%文本数: {front_30_count} (ID: 1-{front_30_count})")
        print(f"   📊 中40%文本数: {len(middle_40_ids)} (ID: {middle_40_start+1}-{middle_40_end})")
        print(f"   📊 后30%文本数: {len(back_30_ids)} (ID: {back_30_start+1}-{total_texts})")
        
        # 从JSON结果中提取被引用的文本ID
        referenced_ids = set()
        topics = json_data.get('topics', [])
        
        for topic in topics:
            if isinstance(topic, dict):
                source_titles = topic.get('source_titles_with_ids', [])
                for source in source_titles:
                    if isinstance(source, dict) and 'id' in source:
                        referenced_ids.add(str(source['id']))
        
        # 计算各区间文本在结果中的出现情况
        front_30_referenced = front_30_ids.intersection(referenced_ids)
        middle_40_referenced = middle_40_ids.intersection(referenced_ids)
        back_30_referenced = back_30_ids.intersection(referenced_ids)

        # 计算各区间的引用比例
        front_30_ratio = len(front_30_referenced) / front_30_count if front_30_count > 0 else 0
        middle_40_ratio = len(middle_40_referenced) / len(middle_40_ids) if len(middle_40_ids) > 0 else 0
        back_30_ratio = len(back_30_referenced) / len(back_30_ids) if len(back_30_ids) > 0 else 0
        
        # 计算位置偏好指数
        front_vs_middle_bias = front_30_ratio / (middle_40_ratio + 0.001)  # 前端vs中间偏好
        back_vs_middle_bias = back_30_ratio / (middle_40_ratio + 0.001)   # 后端vs中间偏好
        front_vs_back_bias = front_30_ratio / (back_30_ratio + 0.001)     # 前端vs后端偏好

        # 计算中间忽视程度
        middle_neglect_score = 1 - middle_40_ratio  # 中间忽视比例（越高越严重）

        # 计算整体位置均衡性（标准差越小越均衡）
        position_balance = np.std([front_30_ratio, middle_40_ratio, back_30_ratio])

        results = {
            "metric_name": "输入文本位置偏好分析（前30% | 中40% | 后30%）",
            "total_texts": total_texts,
            "segment_counts": {
                "front_30_count": front_30_count,
                "middle_40_count": len(middle_40_ids),
                "back_30_count": len(back_30_ids)
            },
            "segment_referenced": {
                "front_30_referenced": len(front_30_referenced),
                "middle_40_referenced": len(middle_40_referenced),
                "back_30_referenced": len(back_30_referenced)
            },
            "segment_ratios": {
                "front_30_ratio": round(front_30_ratio, 4),
                "middle_40_ratio": round(middle_40_ratio, 4),
                "back_30_ratio": round(back_30_ratio, 4)
            },
            "bias_indices": {
                "front_vs_middle": round(front_vs_middle_bias, 4),
                "back_vs_middle": round(back_vs_middle_bias, 4),
                "front_vs_back": round(front_vs_back_bias, 4)
            },
            "neglect_scores": {
                "middle_neglect": round(middle_neglect_score, 4),
                "position_balance": round(position_balance, 4)
            },
            "total_referenced": len(referenced_ids),
            "overall_reference_ratio": round(len(referenced_ids) / total_texts, 4),
            "position_analysis": {
                "front_bias": "高" if front_30_ratio > middle_40_ratio * 1.5 else "中" if front_30_ratio > middle_40_ratio else "低",
                "middle_neglect": "严重" if middle_40_ratio < front_30_ratio * 0.5 else "中等" if middle_40_ratio < front_30_ratio * 0.8 else "轻微",
                "back_bias": "高" if back_30_ratio > middle_40_ratio * 1.5 else "中" if back_30_ratio > middle_40_ratio else "低",
                "overall_balance": "均衡" if position_balance < 0.1 else "轻微偏好" if position_balance < 0.2 else "明显偏好"
            }
        }
        
        # 输出结果
        print(f"\n🔍 输入文本位置偏好分析结果:")
        print(f"   📊 总文本数: {results['total_texts']}")
        print(f"   📊 前30%文本引用率: {results['segment_ratios']['front_30_ratio']:.2%} ({results['segment_referenced']['front_30_referenced']}/{results['segment_counts']['front_30_count']})")
        print(f"   📊 中40%文本引用率: {results['segment_ratios']['middle_40_ratio']:.2%} ({results['segment_referenced']['middle_40_referenced']}/{results['segment_counts']['middle_40_count']})")
        print(f"   📊 后30%文本引用率: {results['segment_ratios']['back_30_ratio']:.2%} ({results['segment_referenced']['back_30_referenced']}/{results['segment_counts']['back_30_count']})")
        print(f"   📊 中间忽视得分: {results['neglect_scores']['middle_neglect']:.4f} (越低越好)")
        print(f"   📊 位置均衡性: {results['neglect_scores']['position_balance']:.4f} (越低越均衡)")
        print(f"   📊 整体评估: {results['position_analysis']['overall_balance']}")

        # 详细分析
        if front_30_ratio > middle_40_ratio * 1.5:
            print(f"   ⚠️ 检测到明显的前端偏好：前30%文本引用率显著高于中间部分")

        if back_30_ratio > middle_40_ratio * 1.5:
            print(f"   ⚠️ 检测到明显的后端偏好：后30%文本引用率显著高于中间部分")

        if middle_40_ratio < front_30_ratio * 0.5 or middle_40_ratio < back_30_ratio * 0.5:
            print(f"   ⚠️ 检测到中间文本忽视现象：中间部分引用率过低")

        # 偏好指数分析
        print(f"\n📈 位置偏好指数:")
        print(f"   前端vs中间: {results['bias_indices']['front_vs_middle']:.2f}")
        print(f"   后端vs中间: {results['bias_indices']['back_vs_middle']:.2f}")
        print(f"   前端vs后端: {results['bias_indices']['front_vs_back']:.2f}")
        
        return results
        
    except FileNotFoundError as e:
        return {"error": f"文件未找到: {str(e)}"}
    except pd.errors.EmptyDataError:
        return {"error": "CSV文件为空"}
    except json.JSONDecodeError:
        return {"error": f"JSON文件格式错误: {json_file}"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "position_bias_analysis_results.json"):
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
    print("🚀 输入文本位置偏好分析（前30% | 中40% | 后30%）评估")
    print("="*70)

    # 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ CSV文件不存在: {INPUT_FILE}")
        return

    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON文件不存在: {JSON_FILE}")
        return

    # 运行分析
    results = analyze_position_bias_ratio(INPUT_FILE, JSON_FILE)
    
    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return
    
    # 保存结果
    save_results(results)
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
