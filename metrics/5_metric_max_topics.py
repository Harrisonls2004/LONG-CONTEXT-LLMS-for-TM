#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指令约束格式下主题数量上限评估指标

评估在满足指令要求下，LLM 可生成的最大主题数量
这个任务需要更改提示词，避免生成固定数量主题，而是提示模型生成尽可能多的主题数量
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import sys
import time

# 添加父目录到路径以导入LLM客户端
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from topic_analyzer.topic_analyzer import OpenRouterClient
except ImportError:
    print("❌ 无法导入LLM客户端，请检查路径")
    sys.exit(1)

# 配置区域
INPUT_FILE = "data/NYT_sampled.csv"
API_KEY = "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"  # 请替换为实际API密钥
MODEL = "qwen/qwen3-235b-a22b:free"
MAX_ATTEMPTS = 3  # 最大尝试次数
SAMPLE_SIZE = 200  # 用于测试的样本大小 - 增加以支持更多主题生成

def measure_max_topic_count(csv_file: str, api_key: str, model: str = MODEL, max_attempts: int = MAX_ATTEMPTS) -> Dict[str, Any]:
    """
    测量指令约束下的最大主题数量
    
    Args:
        csv_file (str): 输入CSV文件路径
        api_key (str): API密钥
        model (str): 使用的模型
        max_attempts (int): 最大尝试次数
        
    Returns:
        Dict: 测量结果
    """
    print("🔢 测量指令约束格式下主题数量上限...")
    print(f"   📁 CSV文件: {csv_file}")
    print(f"   🤖 模型: {model}")
    print(f"   🔄 最大尝试次数: {max_attempts}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        if 'text' not in df.columns:
            return {"error": "CSV文件必须包含'text'列"}
        
        # 随机采样文本用于测试
        sample_texts = df['text'].sample(n=min(SAMPLE_SIZE, len(df))).tolist()
        
        # 初始化LLM客户端
        llm_client = OpenRouterClient(api_key=api_key, model=model)
        
        # 构建测试文本
        combined_text = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(sample_texts)])

        print(f"   📊 输入文本统计:")
        print(f"      文本总长度: {len(combined_text):,} 字符")
        print(f"      文档数量: {len(sample_texts)} 篇")

        # 第一步：让LLM自己估计能生成多少个主题
        print(f"   🤖 让LLM自己估计主题数量...")

        estimation_prompt = f"""
Please analyze the following {len(sample_texts)} news texts and estimate how many distinct topics you can generate from them.

Text content:
{combined_text}

Please provide your estimation in JSON format:
{{
    "estimated_topics_count": your_estimated_number,
    "confidence_level": "high/medium/low",
    "reasoning": "Brief explanation of your estimation"
}}

Be honest about your capabilities and provide a realistic estimate based on the content diversity and your analysis capacity.
"""

        try:
            estimation_response = llm_client.call_llm(estimation_prompt, max_tokens=500, temperature=0.1)

            print(f"   📝 LLM原始响应: {estimation_response[:200]}...")

            # 解析LLM的估计 - 更健壮的JSON解析
            import re
            import json

            llm_estimated_count = 0
            confidence_level = 'unknown'
            reasoning = 'No reasoning provided'

            # 尝试多种JSON解析方法
            json_patterns = [
                r'\{[^{}]*"estimated_topics_count"[^{}]*\}',  # 简单JSON
                r'\{.*?"estimated_topics_count".*?\}',        # 更宽松的匹配
                r'\{.*\}',                                    # 最宽松的匹配
            ]

            parsed = False
            for pattern in json_patterns:
                json_match = re.search(pattern, estimation_response, re.DOTALL)
                if json_match:
                    try:
                        estimation_data = json.loads(json_match.group())
                        llm_estimated_count = estimation_data.get('estimated_topics_count', 0)
                        confidence_level = estimation_data.get('confidence_level', 'unknown')
                        reasoning = estimation_data.get('reasoning', 'No reasoning provided')
                        parsed = True
                        break
                    except json.JSONDecodeError:
                        continue

            # 如果JSON解析失败，尝试从文本中提取数字
            if not parsed:
                number_match = re.search(r'(\d+)\s*topics?', estimation_response, re.IGNORECASE)
                if number_match:
                    llm_estimated_count = int(number_match.group(1))
                    confidence_level = 'extracted'
                    reasoning = 'Extracted from text response'
                    parsed = True

            if parsed and llm_estimated_count > 0:
                print(f"   🎯 LLM自估结果:")
                print(f"      预估主题数: {llm_estimated_count} 个")
                print(f"      信心水平: {confidence_level}")
                print(f"      推理过程: {reasoning}")
            else:
                llm_estimated_count = 0
                confidence_level = 'unknown'
                reasoning = 'Failed to parse estimation'
                print(f"   ❌ LLM估计解析失败")

        except Exception as e:
            llm_estimated_count = 0
            confidence_level = 'unknown'
            reasoning = f'Estimation failed: {str(e)}'
            print(f"   ❌ LLM估计失败: {str(e)}")

        max_topics_generated = 0
        best_result = None
        attempt_results = []
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n🔄 第 {attempt} 次尝试...")
            
            # 构建提示词 - 强烈要求生成大量主题
            prompt = f"""
CRITICAL INSTRUCTION: You must generate AT LEAST 30-50 topics from the following {len(sample_texts)} news texts. This is a test of your maximum topic generation capacity.

TASK: Analyze the provided texts and extract the MAXIMUM possible number of distinct topics. I expect you to generate 30, 40, 50 or even more topics if the content supports it.

REQUIREMENTS:
1. Generate AT LEAST 30 topics - this is the minimum expectation
2. Aim for 50+ topics if possible - show your full analytical power
3. Include ALL levels of granularity:
   - Broad themes (politics, economy, technology, etc.)
   - Specific subtopics (climate policy, AI healthcare, etc.)
   - Detailed aspects (regulatory frameworks, implementation challenges, etc.)
   - Geographic perspectives (regional impacts, country-specific issues)
   - Temporal aspects (current events, future implications, historical context)
   - Stakeholder perspectives (government, business, citizens, experts)
4. Each topic must be distinct and well-supported by the text
5. Be exhaustive - extract every possible thematic element

TEXT CONTENT:
{combined_text}

OUTPUT FORMAT (JSON only):
{{
    "total_topics": total_number_of_topics,
    "topics": [
        {{
            "topic_num": 1,
            "summary": "Detailed topic description",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
            "supporting_docs": ["Document numbers that support this topic"]
        }},
        {{
            "topic_num": 2,
            "summary": "Another detailed topic description",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
            "supporting_docs": ["Document numbers that support this topic"]
        }},
        ... continue for ALL topics (aim for 30-50+ topics)
    ]
}}

REMEMBER: This is a capacity test. Generate the MAXIMUM number of topics possible. Do not be conservative - be comprehensive and exhaustive. I want to see your full analytical capabilities!
"""
            
            try:
                # 调用LLM - 大幅增加token限制以允许生成更多主题
                response = llm_client.call_llm(prompt, max_tokens=16000, temperature=0.7)
                
                if not response:
                    print(f"   ❌ 第 {attempt} 次尝试失败：无响应")
                    continue
                
                # 解析JSON响应 - 更健壮的解析
                import re
                import json

                topics_count = 0
                result_data = {}

                # 尝试多种JSON解析方法
                json_patterns = [
                    r'\{[^{}]*"topics"[^{}]*\[.*?\][^{}]*\}',  # 包含topics数组的JSON
                    r'\{.*?"topics".*?\}',                     # 更宽松的匹配
                    r'\{.*\}',                                 # 最宽松的匹配
                ]

                parsed = False
                for pattern in json_patterns:
                    json_match = re.search(pattern, response, re.DOTALL)
                    if json_match:
                        try:
                            result_data = json.loads(json_match.group())
                            topics_count = len(result_data.get('topics', []))
                            parsed = True
                            break
                        except json.JSONDecodeError:
                            continue

                # 如果JSON解析失败，尝试计算Topic关键词出现次数
                if not parsed or topics_count == 0:
                    topic_matches = re.findall(r'"topic[_\s]*(?:num|id|number)"?\s*:\s*\d+', response, re.IGNORECASE)
                    if topic_matches:
                        topics_count = len(topic_matches)
                        result_data = {"topics": [{"topic_num": i+1} for i in range(topics_count)]}
                        parsed = True

                if parsed and topics_count > 0:
                    attempt_result = {
                        "attempt": attempt,
                        "topics_generated": topics_count,
                        "total_topics_claimed": result_data.get('total_topics', topics_count),
                        "response_length": len(response),
                        "success": True
                    }

                    print(f"   ✅ 第 {attempt} 次尝试成功：生成 {topics_count} 个主题")

                    if topics_count > max_topics_generated:
                        max_topics_generated = topics_count
                        best_result = result_data

                else:
                    attempt_result = {
                        "attempt": attempt,
                        "topics_generated": 0,
                        "error": "无法解析JSON响应",
                        "success": False
                    }
                    print(f"   ❌ 第 {attempt} 次尝试失败：无法解析JSON")
                    print(f"   📝 响应片段: {response[:200]}...")
                
                attempt_results.append(attempt_result)
                
                # 添加延迟避免API限制
                if attempt < max_attempts:
                    time.sleep(2)
                    
            except json.JSONDecodeError as e:
                attempt_result = {
                    "attempt": attempt,
                    "topics_generated": 0,
                    "error": f"JSON解析错误: {str(e)}",
                    "success": False
                }
                attempt_results.append(attempt_result)
                print(f"   ❌ 第 {attempt} 次尝试失败：JSON解析错误")
                
            except Exception as e:
                attempt_result = {
                    "attempt": attempt,
                    "topics_generated": 0,
                    "error": f"调用错误: {str(e)}",
                    "success": False
                }
                attempt_results.append(attempt_result)
                print(f"   ❌ 第 {attempt} 次尝试失败：{str(e)}")
        
        # 计算统计信息
        successful_attempts = [r for r in attempt_results if r['success']]
        topic_counts = [r['topics_generated'] for r in successful_attempts]

        # 计算LLM自估准确性
        actual_avg = np.mean(topic_counts) if topic_counts else 0
        actual_max = max_topics_generated

        # 计算LLM自估的准确性
        if llm_estimated_count > 0:
            estimation_accuracy_ratio = round(actual_avg / llm_estimated_count, 2)
            max_accuracy_ratio = round(actual_max / llm_estimated_count, 2)

            if 0.8 <= estimation_accuracy_ratio <= 1.2:
                estimation_quality = "准确"
            elif estimation_accuracy_ratio < 0.8:
                estimation_quality = "高估"
            else:
                estimation_quality = "低估"
        else:
            estimation_accuracy_ratio = 0
            max_accuracy_ratio = 0
            estimation_quality = "无估计"

        estimation_accuracy = {
            "llm_estimated_count": llm_estimated_count,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "avg_accuracy_ratio": estimation_accuracy_ratio,
            "max_accuracy_ratio": max_accuracy_ratio,
            "estimation_quality": estimation_quality
        }

        results = {
            "metric_name": "指令约束格式下主题数量上限",
            "model": model,
            "sample_size": len(sample_texts),
            "input_statistics": {
                "text_length": len(combined_text),
                "document_count": len(sample_texts)
            },
            "llm_self_estimation": {
                "estimated_topics_count": llm_estimated_count,
                "confidence_level": confidence_level,
                "reasoning": reasoning
            },
            "generation_results": {
                "max_attempts": max_attempts,
                "successful_attempts": len(successful_attempts),
                "max_topics_generated": max_topics_generated,
                "average_topics": round(actual_avg, 2),
                "min_topics": min(topic_counts) if topic_counts else 0,
                "topic_generation_stability": round(np.std(topic_counts), 2) if len(topic_counts) > 1 else 0,
                "success_rate": round(len(successful_attempts) / max_attempts, 2)
            },
            "estimation_accuracy": estimation_accuracy,
            "attempt_details": attempt_results,
            "best_result_sample": best_result.get('topics', [])[:5] if best_result else [],  # 显示前5个主题作为样例
            "evaluation": {
                "generation_capacity": "优秀" if max_topics_generated >= 20 else "良好" if max_topics_generated >= 10 else "一般" if max_topics_generated >= 5 else "较低",
                "stability": "稳定" if len(topic_counts) > 1 and np.std(topic_counts) < 3 else "中等" if len(topic_counts) > 1 and np.std(topic_counts) < 5 else "不稳定",
                "self_estimation_quality": estimation_accuracy["estimation_quality"],
                "overall_assessment": "优秀" if max_topics_generated >= 15 and len(successful_attempts) >= 2 else "良好" if max_topics_generated >= 8 else "需改进"
            }
        }
        
        # 输出结果
        print(f"\n🔢 主题数量上限测量结果:")
        print(f"   📊 输入统计:")
        print(f"      文本长度: {results['input_statistics']['text_length']:,} 字符")
        print(f"      文档数量: {results['input_statistics']['document_count']} 篇")
        print(f"   🤖 LLM自估结果:")
        print(f"      预估主题数: {results['llm_self_estimation']['estimated_topics_count']} 个")
        print(f"      信心水平: {results['llm_self_estimation']['confidence_level']}")
        print(f"      推理过程: {results['llm_self_estimation']['reasoning']}")
        print(f"   📈 实际生成:")
        print(f"      最大主题数: {results['generation_results']['max_topics_generated']}")
        print(f"      平均主题数: {results['generation_results']['average_topics']}")
        print(f"      最少主题数: {results['generation_results']['min_topics']}")
        print(f"   ⚖️ 自估准确性:")
        print(f"      vs平均实际: {results['estimation_accuracy']['avg_accuracy_ratio']}x")
        print(f"      vs最大实际: {results['estimation_accuracy']['max_accuracy_ratio']}x")
        print(f"      自估质量: {results['estimation_accuracy']['estimation_quality']}")
        print(f"   📊 其他指标:")
        print(f"      成功率: {results['generation_results']['success_rate']:.0%}")
        print(f"      生成稳定性: {results['generation_results']['topic_generation_stability']}")
        print(f"      生成能力评估: {results['evaluation']['generation_capacity']}")
        print(f"      整体评估: {results['evaluation']['overall_assessment']}")
        
        return results
        
    except FileNotFoundError:
        return {"error": f"文件未找到: {csv_file}"}
    except pd.errors.EmptyDataError:
        return {"error": "CSV文件为空"}
    except Exception as e:
        return {"error": f"测量过程中发生错误: {str(e)}"}

def save_results(results: Dict[str, Any], output_file: str = "max_topics_results.json"):
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
    print("🚀 指令约束格式下主题数量上限评估")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ CSV文件不存在: {INPUT_FILE}")
        return
    
    # 检查API密钥
    if not API_KEY or API_KEY == "your_api_key_here":
        print("❌ 请在配置区域设置有效的API密钥")
        return
    
    # 运行测量
    results = measure_max_topic_count(INPUT_FILE, API_KEY, MODEL, MAX_ATTEMPTS)
    
    if "error" in results:
        print(f"❌ 测量失败: {results['error']}")
        return
    
    # 保存结果
    save_results(results)
    
    print("\n🎉 测量完成！")

if __name__ == "__main__":
    main()
