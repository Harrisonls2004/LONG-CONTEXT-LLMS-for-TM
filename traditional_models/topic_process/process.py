#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYT数据集主题层次化处理脚本
使用LLM为每行数据生成二/三级主题，并添加到CSV文件的新列中
"""

import os
import csv
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

# 导入配置
try:
    from config import (
        RECOMMENDED_MODELS, DEFAULT_MODEL, RATE_LIMIT_INTERVAL,
        MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
        PROMPT_TEMPLATE_FOCUSED, PROMPT_TEMPLATE_MODERATE, PROMPT_TEMPLATE_STRICT, PROMPT_TEMPLATE_MAXIMUM,
        TEXT_PREVIEW_LENGTH, TOPIC_SEPARATOR,
        ERROR_MARKER, DEFAULT_ENCODING_OPTIONS
    )
except ImportError:
    # 如果没有config.py，使用默认配置
    RECOMMENDED_MODELS = [
        "qwen/qwen3-14b:free",
        "qwen/qwen3-coder:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1-0528:free",
        "deepseek/deepseek-chat-v3.1:free"
    ]
    DEFAULT_MODEL = "qwen/qwen3-14b:free"
    RATE_LIMIT_INTERVAL = 2.0
    MAX_RETRIES = 3
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_TEMPERATURE = 0.3
    TEXT_PREVIEW_LENGTH = 1000
    TOPIC_SEPARATOR = '; '
    ERROR_MARKER = 'ERROR'
    DEFAULT_ENCODING_OPTIONS = ['utf-8', 'gbk', 'latin-1', 'cp1252']

    # 默认提示词模板（英文版本）
    PROMPT_TEMPLATE = """You are a professional news topic analysis expert. Please generate a detailed topic hierarchy for the following news article.

**Article Information:**
Title: {title}
Primary Topic: {primary_topic}
Keywords: {keywords}
Content: {text_preview}...

**Requirements:**
1. Based on the article content, generate as many secondary and tertiary topics as possible
2. Do not limit the number of topics - generate all relevant topics
3. Topics should be specific, accurate, and reflect different aspects of the article
4. Secondary topics should be subdivisions of the primary topic
5. Tertiary topics should be further subdivisions of secondary topics

**Output Format (strictly follow JSON format):**
```json
{{
    "secondary_topics": [
        "Specific secondary topic 1",
        "Specific secondary topic 2",
        "Specific secondary topic 3"
    ],
    "tertiary_topics": [
        "Specific tertiary topic 1",
        "Specific tertiary topic 2",
        "Specific tertiary topic 3",
        "Specific tertiary topic 4"
    ]
}}
```

Please ensure:
- The number of generated topics is not fixed, determined by the richness of article content
- Topic descriptions are concise and clear, typically 2-6 words
- All topics are highly relevant to the article content
- Output must be valid JSON format"""


class OpenRouterClient:
    """OpenRouter API客户端 - 参考topic_evaluator.py的实现"""

    def __init__(self, api_key: str = None, model: str = None):
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.model = model or DEFAULT_MODEL
        self.base_url = "https://openrouter.ai/api/v1"

        # 添加速率限制相关属性
        self.last_request_time = 0
        self.min_request_interval = RATE_LIMIT_INTERVAL

        if not self.api_key:
            raise ValueError("未找到OpenRouter API密钥，请设置OPENROUTER_API_KEY环境变量或直接传入api_key参数")

        # 使用OpenAI客户端连接OpenRouter
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/topic-hierarchy-generator",
                "X-Title": "NYT Topic Hierarchy Generator",
            },
        )

        print(f"API密钥已设置: {self.api_key[:15]}...{self.api_key[-12:]}")
        print(f"使用模型: {self.model}")
        print(f"速率限制: 最小请求间隔 {self.min_request_interval} 秒")

    def _rate_limit(self):
        """实施速率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            print(f"速率限制：等待 {sleep_time:.2f} 秒...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def call_llm(self, prompt: str, max_tokens: int = None, temperature: float = None, max_retries: int = None) -> str:
        """调用OpenRouter API，带有重试机制和速率限制"""
        self._rate_limit()

        # 使用配置的默认值
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        temperature = temperature or DEFAULT_TEMPERATURE
        max_retries = max_retries or MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM API调用失败: {str(e)}")
                else:
                    wait_time = (attempt + 1) * 2
                    print(f"第{attempt + 1}次API调用失败，{wait_time}秒后重试: {str(e)}")
                    time.sleep(wait_time)
                    continue


def create_topic_hierarchy_prompt(title: str, primary_topic: str, text: str, keywords: str, strategy: str = "moderate") -> str:
    """创建主题层次化分析的提示词"""

    # 截取文本预览
    text_preview = text[:TEXT_PREVIEW_LENGTH] if len(text) > TEXT_PREVIEW_LENGTH else text

    # 根据策略选择提示词模板
    try:
        if strategy == "strict":
            template = PROMPT_TEMPLATE_STRICT
        elif strategy == "maximum":
            template = PROMPT_TEMPLATE_MAXIMUM
        elif strategy == "moderate":
            template = PROMPT_TEMPLATE_MODERATE
        else:  # focused (默认推荐)
            template = PROMPT_TEMPLATE_FOCUSED
    except NameError:
        # 如果没有导入新的模板，使用默认的
        template = PROMPT_TEMPLATE

    # 使用选择的提示词模板
    prompt = template.format(
        title=title,
        primary_topic=primary_topic,
        keywords=keywords,
        text_preview=text_preview
    )

    return prompt


def parse_topic_response(response: str) -> Dict[str, List[str]]:
    """解析LLM返回的主题层次结构"""
    try:
        # 尝试直接解析JSON
        if response.strip().startswith('{'):
            result = json.loads(response.strip())
        else:
            # 如果响应包含代码块，提取JSON部分
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # 尝试找到第一个{到最后一个}之间的内容
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx+1]
                    result = json.loads(json_str)
                else:
                    raise ValueError("无法找到有效的JSON格式")

        # 验证结果格式
        if not isinstance(result, dict):
            raise ValueError("返回结果不是字典格式")

        secondary_topics = result.get("secondary_topics", [])
        tertiary_topics = result.get("tertiary_topics", [])

        if not isinstance(secondary_topics, list):
            secondary_topics = []
        if not isinstance(tertiary_topics, list):
            tertiary_topics = []

        return {
            "secondary_topics": secondary_topics,
            "tertiary_topics": tertiary_topics
        }

    except Exception as e:
        print(f"解析LLM响应失败: {str(e)}")
        print(f"原始响应: {response[:200]}...")
        return {
            "secondary_topics": [],
            "tertiary_topics": []
        }


# 推荐的免费模型列表（参考topic_evaluator.py）
RECOMMENDED_MODELS = [
    "qwen/qwen3-14b:free",           # 默认推荐，性能好
    "qwen/qwen3-coder:free",         # 代码理解能力强
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-chat-v3.1:free"
]


def analyze_context_distribution(df: pd.DataFrame, min_text_length: int = 500) -> dict:
    """
    分析数据的上下文长度分布，确定合理的采样策略

    Args:
        df: 原始数据框
        min_text_length: 最小文本长度

    Returns:
        分析结果字典
    """
    print(f"📊 分析数据上下文长度分布...")

    # 计算文本长度
    df['text_length'] = df['text'].astype(str).str.len()
    df_filtered = df[df['text_length'] >= min_text_length].copy()

    # 计算上下文总长度
    def calc_context_length(row):
        title_len = len(str(row.get('title', '')))
        text_len = len(str(row.get('text', '')))
        topic_len = len(str(row.get('topic', '')))
        keywords_len = len(str(row.get('keywords', '')))
        return title_len + text_len + topic_len + keywords_len

    df_filtered['context_length'] = df_filtered.apply(calc_context_length, axis=1)

    # 分析分布
    context_lengths = df_filtered['context_length']

    # 计算分位数
    percentiles = [50, 70, 80, 90, 95, 99]
    percentile_values = {}
    for p in percentiles:
        percentile_values[p] = context_lengths.quantile(p/100)

    # 确定采样策略
    total_count = len(df_filtered)

    # 采样策略：选择前10-20%的最大上下文数据
    if total_count >= 1000:
        # 大数据集：选择前10%
        sample_ratio = 0.10
        strategy = "大数据集策略"
    elif total_count >= 500:
        # 中等数据集：选择前15%
        sample_ratio = 0.15
        strategy = "中等数据集策略"
    else:
        # 小数据集：选择前20%
        sample_ratio = 0.20
        strategy = "小数据集策略"

    suggested_sample_size = max(50, int(total_count * sample_ratio))  # 至少50条
    suggested_sample_size = min(suggested_sample_size, 500)  # 最多500条

    analysis = {
        'total_count': total_count,
        'min_length': int(context_lengths.min()),
        'max_length': int(context_lengths.max()),
        'mean_length': int(context_lengths.mean()),
        'median_length': int(context_lengths.median()),
        'percentiles': percentile_values,
        'suggested_sample_size': suggested_sample_size,
        'sample_ratio': sample_ratio,
        'strategy': strategy
    }

    return analysis


def sample_max_context_data(df: pd.DataFrame, min_text_length: int = 500, auto_determine_size: bool = True, manual_sample_size: int = None) -> pd.DataFrame:
    """
    从数据集中采样最大上下文长度的数据

    Args:
        df: 原始数据框
        min_text_length: 最小文本长度
        auto_determine_size: 是否自动确定采样数量
        manual_sample_size: 手动指定的采样数量（当auto_determine_size=False时使用）

    Returns:
        采样后的数据框
    """
    print(f"📊 开始采样最大上下文数据...")
    print(f"   原始数据量: {len(df):,}")

    # 计算文本长度（不过滤，只用于统计）
    df['text_length'] = df['text'].astype(str).str.len()
    df_filtered = df.copy()  # 不过滤任何数据
    print(f"   数据量: {len(df_filtered):,} (不限制文本长度)")

    if len(df_filtered) == 0:
        print("❌ 没有数据")
        return df.head(0)

    # 计算每行的上下文总长度
    def calc_context_length(row):
        title_len = len(str(row.get('title', '')))
        text_len = len(str(row.get('text', '')))
        topic_len = len(str(row.get('topic', '')))
        keywords_len = len(str(row.get('keywords', '')))
        return title_len + text_len + topic_len + keywords_len

    print("📏 计算上下文总长度...")
    df_filtered['context_length'] = df_filtered.apply(calc_context_length, axis=1)

    # 确定采样数量
    if auto_determine_size:
        analysis = analyze_context_distribution(df_filtered, min_text_length)
        sample_size = analysis['suggested_sample_size']

        print(f"\n📈 数据分析结果:")
        print(f"   采样策略: {analysis['strategy']}")
        print(f"   建议采样数量: {sample_size} 条 ({analysis['sample_ratio']*100:.0f}%)")
        print(f"   上下文长度分布:")
        print(f"     最小: {analysis['min_length']:,}")
        print(f"     中位数: {analysis['median_length']:,}")
        print(f"     平均: {analysis['mean_length']:,}")
        print(f"     最大: {analysis['max_length']:,}")
        print(f"   分位数分析:")
        for p, v in analysis['percentiles'].items():
            print(f"     {p}%分位数: {v:,.0f}")
    else:
        sample_size = manual_sample_size or 100
        print(f"   手动指定采样数量: {sample_size}")

    # 按上下文长度排序，取最大的
    df_sorted = df_filtered.sort_values('context_length', ascending=False)
    actual_sample_size = min(sample_size, len(df_sorted))
    df_sample = df_sorted.head(actual_sample_size).copy()

    # 统计信息
    print(f"\n� 最终采样统计:")
    print(f"   实际采样数量: {len(df_sample)}")
    print(f"   采样比例: {len(df_sample)/len(df_filtered)*100:.1f}%")
    print(f"   最大上下文长度: {df_sample['context_length'].max():,}")
    print(f"   最小上下文长度: {df_sample['context_length'].min():,}")
    print(f"   平均上下文长度: {df_sample['context_length'].mean():.0f}")
    print(f"   最大文本长度: {df_sample['text_length'].max():,}")
    print(f"   平均文本长度: {df_sample['text_length'].mean():.0f}")

    # 主题分布
    if 'topic' in df_sample.columns:
        topic_counts = df_sample['topic'].value_counts()
        print(f"\n📋 采样数据主题分布:")
        for topic, count in topic_counts.head(8).items():
            print(f"   {topic}: {count} 条")

    # 移除辅助列并重置索引
    df_result = df_sample.drop(['text_length', 'context_length'], axis=1)
    df_result = df_result.reset_index(drop=True)  # 重置索引，确保从0开始连续
    return df_result


def process_nyt_dataset(
    input_file: str,
    api_key: str = None,
    model: str = None,  # 使用配置文件中的默认模型
    strategy: str = "moderate",  # 主题生成策略
    sample_size: int = 1000,  # 采样数量
    min_text_length: int = 500,  # 最小文本长度
    start_row: int = 0,
    max_rows: int = None,
    save_interval: int = 10
) -> None:
    """
    处理NYT数据集，采样最大上下文数据，为每行生成二/三级主题

    Args:
        input_file: 输入CSV文件路径
        api_key: OpenRouter API密钥
        model: 使用的LLM模型
        strategy: 主题生成策略 ("focused", "moderate", "strict", "maximum")
        sample_size: 采样数量（选择前N条最大上下文长度的数据）
        min_text_length: 最小文本长度要求
        start_row: 开始处理的行号（用于断点续传）
        max_rows: 最大处理行数（用于测试）
        save_interval: 每处理多少行保存一次

    采样策略：
        - 选择前sample_size条最大上下文长度的数据
        - 不限制文本长度，确保能采样到足够数量的数据
    """

    # 使用默认模型如果未指定
    model = model or DEFAULT_MODEL

    print(f"=== NYT数据集主题层次化处理开始 ===")
    print(f"📁 输入文件: {input_file}")
    print(f"🤖 使用模型: {model}")
    print(f"🎯 主题生成策略: {strategy}")
    print(f"📊 采样设置: {sample_size}条最大上下文数据 (不限制文本长度)")
    print(f"💾 保存间隔: 每{save_interval}行")

    # 输出文件名（基于采样）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_sampled_with_topics.csv"
    print(f"📁 输出文件: {output_file}")

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return

    # 初始化LLM客户端
    try:
        llm_client = OpenRouterClient(api_key=api_key, model=model)
    except Exception as e:
        print(f"❌ 初始化LLM客户端失败: {str(e)}")
        return

    # 读取CSV文件
    try:
        print(f"📖 正在读取CSV文件...")
        # 尝试不同的编码格式（使用配置）
        encodings = DEFAULT_ENCODING_OPTIONS
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                print(f"✅ 成功读取 {len(df)} 行数据 (编码: {encoding})")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("无法使用常见编码格式读取CSV文件")

        # 进行最大上下文采样（固定1000条）
        print(f"\n" + "="*50)
        df_sampled = sample_max_context_data(df, min_text_length=min_text_length, auto_determine_size=False, manual_sample_size=sample_size)
        if len(df_sampled) == 0:
            print("❌ 采样失败，没有符合条件的数据")
            return

        print(f"✅ 采样完成，将处理 {len(df_sampled)} 条最大上下文数据")
        df = df_sampled  # 使用采样后的数据

        # 检查必要的列是否存在
        required_columns = ['title', 'topic', 'text', 'keywords']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")

    except Exception as e:
        print(f"❌ 读取CSV文件失败: {str(e)}")
        return

    # 重置索引，确保索引连续
    df = df.reset_index(drop=True)

    # 添加主题列（紧接着keywords列后面）
    if 'secondary_topics' not in df.columns:
        # 找到keywords列的位置
        keywords_pos = df.columns.get_loc('keywords')
        # 在keywords后面插入新列
        df.insert(keywords_pos + 1, 'secondary_topics', '')
    if 'tertiary_topics' not in df.columns:
        # 找到secondary_topics列的位置
        secondary_pos = df.columns.get_loc('secondary_topics')
        # 在secondary_topics后面插入tertiary_topics
        df.insert(secondary_pos + 1, 'tertiary_topics', '')

    # 如果max_rows=0，只保存采样数据，不进行主题生成
    if max_rows == 0:
        print("📊 只采样模式：保存采样数据，不生成主题")

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_sampled.csv"
        output_path = os.path.join(os.path.dirname(input_file), output_file)

        # 保存采样数据
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ 采样数据已保存到: {output_path}")
        print(f"📊 采样行数: {len(df):,}")
        return

    # 确定处理范围
    total_rows = len(df)
    end_row = min(start_row + max_rows, total_rows) if max_rows else total_rows

    print(f"📊 处理范围: 第{start_row}行 到 第{end_row-1}行 (共{end_row-start_row}行)")

    # 创建输出文件头部（如果文件不存在）
    if not os.path.exists(output_file):
        print(f"📝 创建输出文件: {output_file}")
        df.head(0).to_csv(output_file, index=False, encoding='utf-8')  # 只写入列头

    # 统计信息
    processed_count = 0
    success_count = 0
    error_count = 0
    start_time = time.time()

    try:
        for idx in range(start_row, end_row):
            row = df.iloc[idx]

            print(f"\n🔄 处理第{idx}行 (进度: {idx-start_row+1}/{end_row-start_row})")
            print(f"   标题: {row['title'][:50]}...")
            print(f"   原始索引: {idx}")  # 显示当前处理的索引

            # 检查是否已经处理过
            if pd.notna(row['secondary_topics']) and row['secondary_topics'].strip():
                print(f"   ⏭️  已处理过，跳过")
                processed_count += 1
                continue

            try:
                # 创建提示词
                prompt = create_topic_hierarchy_prompt(
                    title=str(row['title']),
                    primary_topic=str(row['topic']),
                    text=str(row['text']),
                    keywords=str(row['keywords']),
                    strategy=strategy
                )

                # 调用LLM
                print(f"   🤖 正在调用LLM...")
                llm_response = llm_client.call_llm(prompt, max_tokens=2000, temperature=0.3)

                # 添加请求间隔，避免触发速率限制
                print(f"   ⏱️  等待3秒避免速率限制...")
                time.sleep(3)

                # 解析响应
                topic_hierarchy = parse_topic_response(llm_response)

                # 将主题列表转换为字符串（使用配置的分隔符）
                secondary_topics_str = TOPIC_SEPARATOR.join(topic_hierarchy['secondary_topics'])
                tertiary_topics_str = TOPIC_SEPARATOR.join(topic_hierarchy['tertiary_topics'])

                # 更新DataFrame（使用iloc确保正确的行索引）
                df.iloc[idx, df.columns.get_loc('secondary_topics')] = secondary_topics_str
                df.iloc[idx, df.columns.get_loc('tertiary_topics')] = tertiary_topics_str

                print(f"   ✅ 成功生成主题:")
                print(f"      二级主题({len(topic_hierarchy['secondary_topics'])}个): {secondary_topics_str[:100]}...")
                print(f"      三级主题({len(topic_hierarchy['tertiary_topics'])}个): {tertiary_topics_str[:100]}...")

                # 立即保存这一行到文件
                row_to_save = df.iloc[idx:idx+1]  # 获取当前行
                if idx == start_row:
                    # 第一行：覆盖文件（包含列头）
                    row_to_save.to_csv(output_file, index=False, encoding='utf-8', mode='w')
                    print(f"   💾 已保存第{idx}行到文件（创建新文件）")
                else:
                    # 后续行：追加到文件（不包含列头）
                    row_to_save.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
                    print(f"   💾 已保存第{idx}行到文件")

                success_count += 1

            except Exception as e:
                print(f"   ❌ 处理失败: {str(e)}")
                df.iloc[idx, df.columns.get_loc('secondary_topics')] = ERROR_MARKER
                df.iloc[idx, df.columns.get_loc('tertiary_topics')] = ERROR_MARKER

                # 立即保存错误标记的行
                row_to_save = df.iloc[idx:idx+1]
                if idx == start_row:
                    row_to_save.to_csv(output_file, index=False, encoding='utf-8', mode='w')
                    print(f"   💾 已保存第{idx}行到文件（错误标记）")
                else:
                    row_to_save.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)
                    print(f"   💾 已保存第{idx}行到文件（错误标记）")

                error_count += 1

            processed_count += 1

            # 显示进度统计（每10行显示一次）
            if processed_count % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_row = elapsed_time / processed_count
                remaining_rows = end_row - idx - 1
                estimated_remaining_time = remaining_rows * avg_time_per_row

                print(f"\n📊 进度统计:")
                print(f"   已处理: {processed_count}/{end_row-start_row} 行")
                print(f"   成功: {success_count}, 失败: {error_count}")
                print(f"   平均耗时: {avg_time_per_row:.1f}秒/行")
                print(f"   预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
                print(f"   输出文件: {output_file}")

    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断处理，正在保存当前进度...")

    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")

    finally:
        # 最终统计（不需要再次保存，因为已经实时保存了）
        total_time = time.time() - start_time
        print(f"\n=== 处理完成 ===")
        print(f"📊 最终统计:")
        print(f"   总处理行数: {processed_count}")
        print(f"   成功: {success_count}")
        print(f"   失败: {error_count}")
        print(f"   总耗时: {total_time/60:.1f} 分钟")
        if processed_count > 0:
            print(f"   平均耗时: {total_time/processed_count:.2f} 秒/行" if processed_count > 0 else "")
        print(f"   输出文件: {output_file}")
        print(f"💡 所有数据已实时保存，无需等待最终保存")
        print(f"   原始文件: {input_file} (未修改)")


def main():
    """主函数 - 命令行模式"""
    import argparse

    parser = argparse.ArgumentParser(description='NYT数据集主题层次化处理（直接在原文件上修改）')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径（将直接在此文件上添加新列）')
    parser.add_argument('--api-key', help='OpenRouter API密钥')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='LLM模型名称')
    parser.add_argument('--start-row', type=int, default=0, help='开始处理的行号')
    parser.add_argument('--max-rows', type=int, help='最大处理行数')
    parser.add_argument('--save-interval', type=int, default=10, help='保存间隔')

    args = parser.parse_args()

    process_nyt_dataset(
        input_file=args.input,
        api_key=args.api_key,
        model=args.model,
        start_row=args.start_row,
        max_rows=args.max_rows,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 在这里修改您的设置

    # API密钥（参考topic_evaluator.py的方式）
    API_KEY = "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"  # 实际API密钥

    # 数据文件路径（修改为您的文件路径）
    input_file = "../../data/NYT_Dataset.csv"

    # 使用的模型（从下面选择一个，取消注释即可）
    selected_model = DEFAULT_MODEL  # 使用默认模型: qwen/qwen3-14b:free（更稳定）
    # selected_model = "qwen/qwen3-coder:free"                  # 代码理解能力强（但限流严重）
    # selected_model = "meta-llama/llama-3.3-70b-instruct:free" # 大模型
    # selected_model = "google/gemini-2.0-flash-exp:free"       # Google模型
    # selected_model = "deepseek/deepseek-r1-0528:free"         # DeepSeek模型

    # 主题生成策略选择（重要！控制生成主题数量）
    # "focused"  = 聚焦核心（2-4个二级主题，3-6个三级主题）【推荐】
    # "moderate" = 适度控制（2-5个二级主题，3-8个三级主题）
    # "strict"   = 严格控制（2-3个二级主题，3-5个三级主题）
    # "maximum"  = 最大生成（尽可能多的主题）
    topic_generation_strategy = "focused"

    # 采样设置（从原始数据中选择最大上下文长度的数据）
    sample_size = 14000       # 采样数量：选择前14000条最大上下文长度的数据
    min_text_length = 0       # 不限制最小文本长度，保证能采样到14000条

    # 处理设置
    process_all_data = False   # True=处理全部数据, False=只处理部分数据
    test_rows = 0             # 如果process_all_data=False，处理多少行（0=只采样不生成主题）
    save_every = 50           # 每处理多少行保存一次

    # ==================== 配置区域结束 ====================

    print("🚀 开始处理NYT数据集...")
    print("💡 提示：使用脚本内设置的API密钥")
    print(f"� 使用模型: {selected_model}")
    print(f"📋 推荐的免费模型: {', '.join(RECOMMENDED_MODELS[:3])}")
    print(f"📁 数据文件: {input_file}")
    print(f"📊 采样设置: {sample_size}条最大上下文数据 (不限制文本长度)")
    print(f"⚙️  处理模式: {'全部采样数据' if process_all_data else f'测试模式({test_rows}行)'}")

    # 显示主题生成策略
    strategy_descriptions = {
        "focused": "聚焦核心（2-4个二级主题，3-6个三级主题）",
        "moderate": "适度控制（2-5个二级主题，3-8个三级主题）",
        "strict": "严格控制（2-3个二级主题，3-5个三级主题）",
        "maximum": "最大生成（尽可能多的主题）"
    }
    print(f"🎯 主题生成策略: {topic_generation_strategy} - {strategy_descriptions.get(topic_generation_strategy, '未知策略')}")

    print(f"🔑 API密钥: {API_KEY[:20]}...")  # 只显示前20个字符

    # 检查API密钥
    if API_KEY and API_KEY != "your_openrouter_api_key_here":
        api_key = API_KEY
        print("✅ 使用脚本中设置的API密钥")
    else:
        print("\n❌ 错误：请设置您的API密钥")
        print("请修改脚本中的配置：")
        print('API_KEY = "your_actual_api_key_here"  # 替换为您的实际API密钥')
        print("\n如何获取API密钥：")
        print("1. 访问 https://openrouter.ai/")
        print("2. 注册并登录账号")
        print("3. 在控制台获取API密钥")
        print("4. 将密钥填入上面的API_KEY变量中")
        exit(1)

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"\n❌ 错误：输入文件不存在: {input_file}")
        exit(1)

    if process_all_data:
        # 询问用户是否要处理全部采样数据
        print(f"\n⚠️  即将从原始数据集中采样 {sample_size} 条最大上下文数据并生成主题")
        print("如果只想测试，请修改脚本中的 process_all_data = False")
        response = input("确认开始采样和处理？(y/n): ").lower().strip()

        if response != 'y':
            print("❌ 用户取消处理")
            exit(0)

        max_rows_to_process = None
        print("🚀 开始采样和处理...")
    else:
        max_rows_to_process = test_rows
        print(f"🧪 测试模式：采样后只处理前 {test_rows} 行")

    try:
        # 开始处理
        process_nyt_dataset(
            input_file=input_file,
            api_key=api_key,
            model=selected_model,
            strategy=topic_generation_strategy,
            sample_size=sample_size,
            min_text_length=min_text_length,
            start_row=0,
            max_rows=max_rows_to_process,
            save_interval=save_every
        )

        if process_all_data:
            print("\n🎉 采样和主题生成完成！")
        else:
            print(f"\n🎉 测试处理完成！处理了 {test_rows} 行采样数据")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        exit(1)