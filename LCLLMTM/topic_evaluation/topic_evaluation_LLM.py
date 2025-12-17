#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM主题评估模块
整合 prompt_list.py、topic_analyzer.py 和 topic_evaluation.py 的功能
使用大型语言模型对主题词列表进行自动化评估
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import time


class OpenRouterClient:
    """OpenRouter API客户端 - 基于 topic_analyzer.py"""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("未找到OpenRouter API密钥，请设置OPENROUTER_API_KEY环境变量或直接传入api_key参数")

        print(f"API密钥已设置: {self.api_key[:15]}...{self.api_key[-12:]}")
    
    def call_llm(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """调用OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Topic Evaluation LLM"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API返回格式异常: {result}")

        except requests.exceptions.RequestException as e:
            if "401" in str(e):
                raise Exception(f"OpenRouter API调用失败: 401 Unauthorized - API密钥无效或已过期")
            elif "402" in str(e):
                raise Exception(f"OpenRouter API调用失败: 402 Payment Required - 账户余额不足")
            elif "429" in str(e):
                raise Exception(f"OpenRouter API调用失败: 429 Too Many Requests - 请求过于频繁，请稍后重试")
            else:
                raise Exception(f"OpenRouter API调用失败: {str(e)}")
        except Exception as e:
            raise Exception(f"处理响应失败: {str(e)}")


def create_evaluation_prompt(topic_keywords: List[str]) -> str:
    
    prompt = f"""Please evaluate the given topic keyword list based on the following topic quality assessment criteria. For each criterion, provide a score from 1-5 and a brief explanation.

Topic Quality Assessment Criteria:

1. **Coherence (语义一致性)**
   Definition: Keywords within the topic should be semantically closely related and collectively describe a coherent theme or related themes.

2. **Conciseness (简洁度)**
   Definition: The topic should not contain irrelevant or meaningless words, such as noise words or semantically redundant terms.

3. **Informativity (信息密度)**
   Definition: The topic description should contain sufficiently specific, meaningful, or valuable information, covering different aspects of the same theme.

Scoring Guidelines:
- 1 point: Poor performance, does not meet the standard requirements
- 2 points: Below average, partially meets some requirements
- 3 points: Average performance, meets basic requirements
- 4 points: Good performance, exceeds basic requirements
- 5 points: Excellent performance, fully meets all standard requirements

Please evaluate the following topic keyword list:
Topic Keywords: {topic_keywords}

Required Response Format (JSON):
{{
  "topic_keywords": {topic_keywords},
  "evaluation": {{
    "coherence": {{
      "score": <1-5>,
      "explanation": "Brief explanation for the score"
    }},
    "conciseness": {{
      "score": <1-5>,
      "explanation": "Brief explanation for the score"
    }},
    "informativity": {{
      "score": <1-5>,
      "explanation": "Brief explanation for the score"
    }}
  }},
  "overall_score": <average of all scores>,
  "overall_assessment": "Overall assessment and recommendations"
}}

Please provide your evaluation in the exact JSON format specified above."""
    
    return prompt


def parse_llm_evaluation(llm_response: str) -> Dict[str, Any]:
    """解析LLM返回的评估结果"""
    try:
        # 尝试直接解析JSON
        result = json.loads(llm_response)
        return result
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass
        
        # 如果仍然失败，返回原始响应
        return {
            "error": "Failed to parse LLM response",
            "raw_response": llm_response
        }


def evaluate_topic_with_retry(
    llm_client: OpenRouterClient,
    topic_num: int,
    keywords: List[str],
    max_retries: int = 3
) -> tuple[Dict[str, Any], str, int]:
    """
    带重试机制的主题评估函数
    
    Args:
        llm_client: LLM客户端
        topic_num: 主题编号
        keywords: 关键词列表
        max_retries: 最大重试次数
    
    Returns:
        tuple: (评估结果, 原始LLM响应, 实际使用的尝试次数)
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🤖 正在调用LLM评估主题 {topic_num} (尝试 {attempt}/{max_retries})...")
            print(f"   关键词: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            
            # 创建评估提示词
            prompt = create_evaluation_prompt(keywords)
            
            # 调用LLM进行评估
            llm_response = llm_client.call_llm(
                prompt, 
                max_tokens=2000, 
                temperature=0.3
            )
            
            print(f"✅ LLM响应已接收，正在解析结果...")
            
            # 解析评估结果
            evaluation = parse_llm_evaluation(llm_response)
            
            # 检查解析是否成功
            if "error" not in evaluation:
                # 解析成功，打印评估结果摘要
                if "evaluation" in evaluation:
                    scores = []
                    for criterion in ["coherence", "conciseness", "informativity"]:
                        if criterion in evaluation["evaluation"] and "score" in evaluation["evaluation"][criterion]:
                            scores.append(f"{criterion}: {evaluation['evaluation'][criterion]['score']}")
                    print(f"📊 主题 {topic_num} 评估完成 - {', '.join(scores)}")
                else:
                    print(f"📊 主题 {topic_num} 评估完成")
                
                return evaluation, llm_response, attempt
            else:
                # 解析失败
                print(f"❌ 主题 {topic_num} 解析失败 (尝试 {attempt}/{max_retries}): {evaluation.get('error', 'Unknown error')}")
                
                if attempt < max_retries:
                    print(f"⏳ 等待5秒后重试...")
                    time.sleep(5)
                else:
                    print(f"❌ 主题 {topic_num} 达到最大重试次数，解析最终失败")
                    return evaluation, llm_response, attempt
                    
        except Exception as e:
            print(f"❌ 主题 {topic_num} 评估出错 (尝试 {attempt}/{max_retries}): {str(e)}")
            
            if attempt < max_retries:
                print(f"⏳ 等待5秒后重试...")
                time.sleep(5)
            else:
                print(f"❌ 主题 {topic_num} 达到最大重试次数，评估最终失败")
                error_evaluation = {
                    "error": f"Failed after {max_retries} attempts: {str(e)}"
                }
                return error_evaluation, None, attempt
    
    # 这里不应该到达，但为了安全起见
    return {"error": "Unexpected error in retry logic"}, None, max_retries


def topic_evaluation_LLM(
    topics_data: List[Dict[str, Any]], 
    output_file: str = "llm_topic_evaluation_results.json",
    api_key: str = None,
    model: str = "moonshotai/kimi-k2:free",
    max_topics: Optional[int] = None
) -> Dict[str, Any]:
    """
    使用LLM对主题词列表进行评估
    
    Args:
        topics_data: 主题数据列表，每个主题包含 'keywords' 字段
        output_file: 输出JSON文件路径
        api_key: OpenRouter API密钥
        model: 使用的LLM模型
        max_topics: 最大评估主题数量（用于控制成本）
    
    Returns:
        包含所有评估结果的字典
    """
    
    print(f"=== LLM主题评估开始 ===")
    print(f"使用模型: {model}")
    print(f"待评估主题数量: {len(topics_data)}")
    
    # 初始化LLM客户端
    try:
        llm_client = OpenRouterClient(api_key=api_key, model=model)
    except Exception as e:
        raise Exception(f"初始化LLM客户端失败: {str(e)}")
    
    # 限制评估主题数量（控制成本）
    if max_topics and len(topics_data) > max_topics:
        topics_data = topics_data[:max_topics]
        print(f"限制评估主题数量为: {max_topics}")
    
    evaluation_results = {
        "metadata": {
            "model": model,
            "total_topics": len(topics_data),
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_criteria": [
                "coherence", "conciseness", "informativity"
            ]
        },
        "topic_evaluations": [],
        "summary_statistics": {}
    }
    
    # 逐个评估主题
    for i, topic in enumerate(topics_data, 1):
        print(f"\n正在评估主题 {i}/{len(topics_data)}...")
        
        # 提取关键词
        keywords = topic.get('keywords', [])
        if not keywords:
            print(f"主题 {i} 没有关键词，跳过")
            continue
        
        topic_num = topic.get('topic_num', i)
        print(f"主题 {topic_num}: {keywords}")
        
        # 使用带重试机制的评估函数
        evaluation, llm_response, attempts_used = evaluate_topic_with_retry(
            llm_client=llm_client,
            topic_num=topic_num,
            keywords=keywords,
            max_retries=3
        )
        
        # 添加主题信息
        topic_evaluation = {
            "topic_num": topic_num,
            "keywords": keywords,
            "summary": topic.get('summary', ''),
            "evaluation": evaluation,
            "raw_llm_response": llm_response,
            "attempts_used": attempts_used
        }
        
        evaluation_results["topic_evaluations"].append(topic_evaluation)
        
        # 添加延迟以避免API限制 - 增加到3秒
        time.sleep(3)
    
    # 计算汇总统计
    evaluation_results["summary_statistics"] = calculate_summary_statistics(
        evaluation_results["topic_evaluations"]
    )
    
    # 保存结果到JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存结果文件失败: {str(e)}")
    
    # 打印评估报告
    print_evaluation_report(evaluation_results)
    
    return evaluation_results


def calculate_summary_statistics(topic_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算汇总统计信息"""
    
    valid_evaluations = [
        eval_data for eval_data in topic_evaluations 
        if "error" not in eval_data.get("evaluation", {})
    ]
    
    if not valid_evaluations:
        return {"error": "No valid evaluations found"}
    
    criteria = ["coherence", "conciseness", "informativity"]
    
    statistics = {
        "total_topics_evaluated": len(valid_evaluations),
        "failed_evaluations": len(topic_evaluations) - len(valid_evaluations),
        "average_scores": {},
        "score_distributions": {},
        "overall_average": 0.0
    }
    
    # 计算各维度平均分
    for criterion in criteria:
        scores = []
        for eval_data in valid_evaluations:
            evaluation = eval_data.get("evaluation", {})
            if criterion in evaluation and "score" in evaluation[criterion]:
                scores.append(evaluation[criterion]["score"])
        
        if scores:
            statistics["average_scores"][criterion] = sum(scores) / len(scores)
            statistics["score_distributions"][criterion] = {
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
    
    # 记录每个维度的平均分
    statistics["dimension_averages"] = {
        "coherence_average": statistics["average_scores"].get("coherence", 0),
        "conciseness_average": statistics["average_scores"].get("conciseness", 0),
        "informativity_average": statistics["average_scores"].get("informativity", 0)
    }
    
    return statistics


def print_evaluation_report(evaluation_results: Dict[str, Any]):
    """打印评估报告"""
    
    print("\n" + "="*80)
    print("📊 LLM主题评估报告")
    print("="*80)
    
    metadata = evaluation_results.get("metadata", {})
    stats = evaluation_results.get("summary_statistics", {})
    
    print(f"\n🔧 评估配置:")
    print(f"   模型: {metadata.get('model', 'Unknown')}")
    print(f"   评估时间: {metadata.get('evaluation_time', 'Unknown')}")
    print(f"   总主题数: {metadata.get('total_topics', 0)}")
    
    print(f"\n📈 评估统计:")
    print(f"   成功评估: {stats.get('total_topics_evaluated', 0)}")
    print(f"   失败评估: {stats.get('failed_evaluations', 0)}")
    
    if "average_scores" in stats:
        print(f"\n📊 各维度平均分:")
        for criterion, score in stats["average_scores"].items():
            print(f"   {criterion.capitalize()}: {score:.2f}/5.0")
    
    print(f"\n💡 评估建议:")
    if "average_scores" in stats and stats["average_scores"]:
        avg_score = sum(stats["average_scores"].values()) / len(stats["average_scores"])
        if avg_score >= 4.0:
            print("   - 主题质量优秀，各维度表现良好")
        elif avg_score >= 3.0:
            print("   - 主题质量良好，可考虑进一步优化")
        elif avg_score >= 2.0:
            print("   - 主题质量一般，建议重点改进低分维度")
        else:
            print("   - 主题质量较差，建议重新生成或大幅优化")
    else:
        print("   - 主题质量较差，建议重新生成或大幅优化")


# 示例使用函数
def example_usage():
    """示例用法"""
    
    # 示例主题数据（通常来自 topic_analyzer.py 的输出）
    sample_topics = [
        {
            "topic_num": 1,
            "summary": "Financial investment and market analysis",
            "keywords": ["investment", "market", "strategy", "risk", "portfolio", "analysis", "finance"]
        },
        {
            "topic_num": 2,
            "summary": "Technology and innovation",
            "keywords": ["technology", "innovation", "digital", "software", "development", "AI", "automation"]
        }
    ]
    
    # 调用评估函数
    try:
        results = topic_evaluation_LLM(
            topics_data=sample_topics,
            output_file="example_evaluation_results.json",
            api_key="your-api-key-here",  # 替换为实际的API密钥
            model="moonshotai/kimi-k2:free",
            max_topics=5  # 限制评估数量以控制成本
        )
        
        print("\n评估完成！")
        return results
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        return None


if __name__ == "__main__":
    # 运行示例
    example_usage()