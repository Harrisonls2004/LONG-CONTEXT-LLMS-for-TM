#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题重复率/多样性评估指标

分析单个JSON文件中主题的重复性和多样性
把主题的summary发给LLM，让LLM自动识别重复的、可合并的主题对
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import sys
import re

# 添加父目录到路径以导入LLM客户端
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from topic_analyzer.topic_analyzer import OpenRouterClient
    print("✅ 成功导入LLM客户端")
except ImportError as e:
    print(f"❌ 无法导入LLM客户端: {e}")
    sys.exit(1)

# 配置区域
JSON_FILE = "C:/Users/1/Desktop/TopMost/LCLLMTM/LLM_response_results/topic_analysis_NYT_sampled_qwen_qwen3_235b_a22bfree.json"
API_KEY = "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"  # 请替换为实际API密钥
MODEL = "moonshotai/kimi-k2:free"

def analyze_topic_diversity(json_file: str, api_key: str, model: str = MODEL) -> Dict[str, Any]:
    """
    分析单个JSON文件中主题的重复率/多样性

    Args:
        json_file (str): 主题JSON文件路径
        api_key (str): API密钥
        model (str): 使用的模型

    Returns:
        Dict: 分析结果
    """
    print("🌈 分析主题重复率/多样性...")
    print(f"   📁 JSON文件: {json_file}")
    print(f"   🤖 模型: {model}")
    
    try:
        # 读取主题JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取主题列表
        all_topics = []
        if 'topics' in data:
            for i, topic in enumerate(data['topics']):
                # 处理不同的主题数据结构
                topic_summary = ""
                topic_id = f"T{i+1}"

                if isinstance(topic, dict):
                    # 方式1: 直接包含summary字段
                    if 'summary' in topic:
                        topic_summary = topic['summary']
                        topic_id = topic.get('topic_num', f"T{i+1}")

                    # 方式2: 嵌套结构，如 {"Topic 1": {"Summary": "...", ...}}
                    else:
                        for key, value in topic.items():
                            if isinstance(value, dict) and 'Summary' in value:
                                topic_summary = value['Summary']
                                topic_id = key
                                break
                            elif isinstance(value, dict) and 'summary' in value:
                                topic_summary = value['summary']
                                topic_id = key
                                break

                if topic_summary.strip():
                    all_topics.append({
                        'id': str(topic_id),
                        'summary': topic_summary.strip(),
                        'index': i+1
                    })
        
        if len(all_topics) < 2:
            return {"error": "主题数量不足，无法进行多样性分析"}

        print(f"   📊 成功提取主题: {len(all_topics)} 个")

        # 初始化LLM客户端
        llm_client = OpenRouterClient(api_key=api_key, model=model)

        # 构建主题列表文本
        topics_text = "\n".join([f"{i+1}. [{topic['id']}] {topic['summary']}" for i, topic in enumerate(all_topics)])

        # 构建LLM分析提示词
        prompt = f"""
请分析以下{len(all_topics)}个主题的重复性和多样性。

主题列表：
{topics_text}

请执行以下分析任务：
1. 识别重复或高度相似的主题对（相似度>75%）- 这些主题可以合并
2. 识别部分重叠的主题对（相似度40-75%）- 这些主题有一定重叠但仍有区别
3. 评估主题的整体多样性程度
4. 计算实际的唯一主题数量

分析标准：
- 重复主题：内容基本相同，可以合并为一个主题
- 部分重叠：有共同元素但侧重点不同
- 多样性评分：0-1之间，1表示完全多样化，0表示完全重复

请以JSON格式返回结果：
{{
    "redundant_pairs": [
        {{
            "topic1_id": "主题1ID",
            "topic1_summary": "主题1摘要",
            "topic2_id": "主题2ID",
            "topic2_summary": "主题2摘要",
            "similarity_score": 0.85,
            "similarity_reason": "相似原因详细描述",
            "can_merge": true
        }}
    ],
    "partial_overlap_pairs": [
        {{
            "topic1_id": "主题1ID",
            "topic2_id": "主题2ID",
            "similarity_score": 0.55,
            "overlap_reason": "重叠原因描述",
            "overlap_areas": ["共同领域1", "共同领域2"]
        }}
    ],
    "unique_topics_count": 实际唯一主题数量,
    "diversity_score": 0到1之间的多样性得分,
    "mergeable_groups": [["可合并主题ID1", "可合并主题ID2", "可合并主题ID3"]],
    "analysis_summary": "详细分析总结，包括主要发现和建议"
}}
"""
        
        print(f"   🤖 调用LLM进行主题多样性分析...")
        
        # 调用LLM分析
        response = llm_client.call_llm(prompt, max_tokens=3000, temperature=0.1)
        
        if not response:
            return {"error": "LLM调用失败，无响应"}
        
        # 解析LLM响应
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
            else:
                return {"error": "无法从LLM响应中提取JSON结果"}
        except json.JSONDecodeError:
            return {"error": "LLM响应的JSON格式错误"}
        
        # 处理分析结果
        redundant_pairs = analysis_result.get("redundant_pairs", [])
        partial_overlap_pairs = analysis_result.get("partial_overlap_pairs", [])
        unique_count = analysis_result.get("unique_topics_count", len(all_topics))
        diversity_score = analysis_result.get("diversity_score", 0.8)
        mergeable_groups = analysis_result.get("mergeable_groups", [])

        # 计算统计指标
        total_pairs = len(all_topics) * (len(all_topics) - 1) // 2
        redundancy_rate = len(redundant_pairs) / len(all_topics) if all_topics else 0
        overlap_rate = len(partial_overlap_pairs) / total_pairs if total_pairs > 0 else 0

        # 计算可合并主题数量
        mergeable_topics_count = sum(len(group) for group in mergeable_groups)
        potential_reduction = len(all_topics) - unique_count

        results = {
            "metric_name": "主题重复率/多样性分析",
            "model": model,
            "input_file": json_file,
            "total_topics": len(all_topics),
            "unique_topics": unique_count,
            "redundant_pairs_count": len(redundant_pairs),
            "partial_overlap_pairs_count": len(partial_overlap_pairs),
            "mergeable_groups_count": len(mergeable_groups),
            "mergeable_topics_count": mergeable_topics_count,
            "potential_reduction": potential_reduction,
            "redundancy_rate": round(redundancy_rate, 4),
            "overlap_rate": round(overlap_rate, 4),
            "diversity_score": round(diversity_score, 4),
            "compression_ratio": round(unique_count / len(all_topics), 4) if all_topics else 1.0,
            "redundant_pairs": redundant_pairs,
            "partial_overlap_pairs": partial_overlap_pairs,
            "mergeable_groups": mergeable_groups,
            "llm_analysis": analysis_result.get("analysis_summary", ""),
            "evaluation": {
                "diversity_quality": "优秀" if diversity_score >= 0.8 else "良好" if diversity_score >= 0.6 else "一般" if diversity_score >= 0.4 else "较差",
                "redundancy_severity": "轻微" if redundancy_rate <= 0.1 else "中等" if redundancy_rate <= 0.3 else "严重",
                "compression_potential": "低" if potential_reduction <= 5 else "中等" if potential_reduction <= 15 else "高",
                "overall_assessment": "优秀" if diversity_score >= 0.7 and redundancy_rate <= 0.2 else "良好" if diversity_score >= 0.5 and redundancy_rate <= 0.4 else "需改进"
            },
            "topics_sample": all_topics[:10]  # 保存前10个主题作为样例
        }
        
        # 输出结果
        print(f"\n🌈 主题重复率/多样性分析结果:")
        print(f"   📊 总主题数: {results['total_topics']}")
        print(f"   📊 唯一主题数: {results['unique_topics']}")
        print(f"   📊 重复主题对: {results['redundant_pairs_count']}")
        print(f"   📊 部分重叠对: {results['partial_overlap_pairs_count']}")
        print(f"   📊 可合并组数: {results['mergeable_groups_count']}")
        print(f"   📊 可合并主题数: {results['mergeable_topics_count']}")
        print(f"   📊 潜在压缩数: {results['potential_reduction']}")
        print(f"   📊 冗余率: {results['redundancy_rate']:.2%}")
        print(f"   📊 多样性得分: {results['diversity_score']:.4f}")
        print(f"   📊 压缩比: {results['compression_ratio']:.4f}")
        print(f"   📊 多样性质量: {results['evaluation']['diversity_quality']}")
        print(f"   📊 冗余严重程度: {results['evaluation']['redundancy_severity']}")
        print(f"   📊 压缩潜力: {results['evaluation']['compression_potential']}")
        print(f"   📊 整体评估: {results['evaluation']['overall_assessment']}")

        # 显示重复主题对示例
        if redundant_pairs:
            print(f"\n🔄 重复主题对示例:")
            for i, pair in enumerate(redundant_pairs[:3], 1):
                print(f"   {i}. {pair['topic1_id']} ↔ {pair['topic2_id']}")
                print(f"      相似度: {pair.get('similarity_score', 'N/A')}")
                print(f"      原因: {pair.get('similarity_reason', 'N/A')}")

        # 显示可合并组示例
        if mergeable_groups:
            print(f"\n📦 可合并主题组示例:")
            for i, group in enumerate(mergeable_groups[:3], 1):
                print(f"   组{i}: {' + '.join(group)}")
        
        return results
        
    except FileNotFoundError as e:
        return {"error": f"文件未找到: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON文件格式错误: {str(e)}"}
    except Exception as e:
        return {"error": f"分析过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "topic_diversity_results.json"):
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
    print("🚀 主题重复率/多样性评估")
    print("="*60)

    print(f"📁 目标文件: {JSON_FILE}")
    print(f"🤖 使用模型: {MODEL}")

    # 检查文件是否存在
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON文件不存在: {JSON_FILE}")
        return
    else:
        print(f"✅ JSON文件存在")

    # 检查API密钥
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ 请在配置区域设置有效的API密钥")
        return
    else:
        print(f"✅ API密钥已配置: {API_KEY[:20]}...")

    print(f"\n🔄 开始分析...")

    # 运行分析
    results = analyze_topic_diversity(JSON_FILE, API_KEY, MODEL)

    if "error" in results:
        print(f"❌ 分析失败: {results['error']}")
        return

    # 保存结果
    save_results(results)

    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
