#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主题分析器 - 使用OpenRouter API调用各种LLM模型
支持通过OpenRouter访问GPT、Claude、Gemini等模型
"""

import os
import json
import re
import csv
import math
import random
from typing import List, Dict, Tuple
import requests
from pathlib import Path
import os
import time

# 强制清除所有代理设置
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
for var in proxy_vars:
    if var in os.environ:
        del os.environ[var]
        print(f"已清除环境变量: {var}")

print("✓ 已强制清除所有代理设置，使用直连模式")
# # 设置代理（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10809'
class OpenRouterClient:
    """OpenRouter API客户端"""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        # 优先使用传入的api_key，如果为None则从环境变量获取
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        # 添加速率限制相关属性
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 最小请求间隔（秒）

        if not self.api_key:
            raise ValueError("未找到OpenRouter API密钥，请设置OPENROUTER_API_KEY环境变量或直接传入api_key参数")

        print(f"API密钥已设置: {self.api_key[:15]}...{self.api_key[-12:]}")  # 显示部分密钥用于调试
        print(f"速率限制: 最小请求间隔 {self.min_request_interval} 秒")
    
    def call_llm(self, prompt: str, max_tokens: int = None, temperature: float = 0, max_retries: int = 3) -> str:
        """调用OpenRouter API，带有重试机制和速率限制"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # 可选：用于统计
            "X-Title": "Text Topic Analyzer"  # 可选：应用名称
        }

        # 如果未指定max_tokens，则使用模型默认的Max Output值
        if max_tokens is None:
            model_info = self.model_context_limits.get(self.model, {})
            max_tokens = model_info.get("Max Output", 20000)*0.9

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning": {
                # "max tokens": 1000,
                "effort": "low",
                "exclude": True,  # Use reasoning but don't include it in the response
                "enabled": False
            }
        }

        # 重试逻辑
        for attempt in range(max_retries):
            try:
                # 实施速率限制
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time

                if time_since_last_request < self.min_request_interval:
                    wait_time = self.min_request_interval - time_since_last_request
                    print(f"速率限制: 等待 {wait_time:.1f} 秒...")
                    time.sleep(wait_time)

                # 强制禁用代理
                proxies = {
                    'http': None,
                    'https': None
                }

                print(f"正在发送API请求 (尝试 {attempt + 1}/{max_retries})...")

                # 更新最后请求时间
                self.last_request_time = time.time()

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=120,  # 增加超时时间
                    proxies=proxies  # 强制不使用代理
                )

                # 检查响应状态
                if response.status_code == 429:
                    # 从响应头获取重试时间
                    retry_after = response.headers.get('Retry-After', '60')
                    try:
                        wait_time = int(retry_after)
                    except:
                        wait_time = 60  # 默认等待60秒

                    if attempt < max_retries - 1:
                        print(f"遇到速率限制 (429)，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"OpenRouter API调用失败: 429 Too Many Requests - 已达到最大重试次数，请稍后再试")

                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    print("✓ API调用成功")
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"API返回格式异常: {result}")

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    # 对于网络错误，等待较短时间后重试
                    if "429" not in str(e):
                        wait_time = min(10 * (attempt + 1), 30)  # 递增等待时间，最多30秒
                        print(f"网络错误，等待 {wait_time} 秒后重试: {str(e)}")
                        time.sleep(wait_time)
                        continue

                # 提供更详细的错误信息
                if "401" in str(e):
                    raise Exception(f"OpenRouter API调用失败: 401 Unauthorized - API密钥无效或已过期。请检查: 1) API密钥是否正确 2) 账户是否有余额 3) API密钥是否有权限访问该模型")
                elif "402" in str(e):
                    raise Exception(f"OpenRouter API调用失败: 402 Payment Required - 账户余额不足，请前往 https://openrouter.ai/ 充值")
                elif "429" in str(e):
                    raise Exception(f"OpenRouter API调用失败: 429 Too Many Requests - 请求过于频繁，已重试 {max_retries} 次，请稍后重试")
                else:
                    raise Exception(f"OpenRouter API调用失败: {str(e)}")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"处理响应失败，等待 {wait_time} 秒后重试: {str(e)}")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"处理响应失败: {str(e)}")

        raise Exception("API调用失败: 已达到最大重试次数")


class TopicAnalyzer:
    """文本主题分析器"""

    def __init__(self, llm_client: OpenRouterClient):
        self.llm_client = llm_client
        self.token_limit = 30000
        self.avg_chars_per_token = 3.5  # 平均每个token的字符数
        
        

        # 模型最大上下文长度配置（单位：tokens）
        self.model_context_limits = { 
            # "openai/gpt-5-mini": {"Total Context": 400000, "Max Output": 128000}, # 集成了推理模型不一定使用 
            "openai/gpt-5-mini": {"Total Context": 300000, "Max Output": 128000}, # 集成了推理模型不一定使用 ，原上下文长度无法输入
            "openai/gpt-5": {"Total Context": 400000, "Max Output": 128000}, # 集成了推理模型不一定使用 
            "openai/gpt-4o-2024-11-20": {"Total Context":  128000, "Max Output": 16400}, 
            "anthropic/claude-opus-4.1": {"Total Context": 200000, "Max Output": 32000}, # 推理模式
            "anthropic/claude-sonnet-4": {"Total Context": 200000, "Max Output": 64000}, # 推理模式
            "google/gemini-pro-1.5": {"Total Context": 2000000, "Max Output": 8200},   # 推理模式
            "google/gemini-2.5-flash-lite": {"Total Context": 1050000, "Max Output": 65500}, # 推理模式
            # "google/gemini-2.5-pro-preview": {"Total Context": 1050000, "Max Output": 65500}, # 推理模式 
            "google/gemini-2.5-pro-preview": {"Total Context": 750000, "Max Output": 65500}, # 推理模式 ，原上下文长度无法输入
            # "google/gemini-2.5-pro-exp-03-25": {"Total Context": 1050000, "Max Output": 65500}, # 已弃用
            "google/gemini-2.0-flash-exp:free": {"Total Context": 1050000, "Max Output": 8200}, # 推理模式
            "meta-llama/llama-4-scout": {"Total Context": 1050000, "Max Output": 1050000}, #高容量多模态语言
            "meta-llama/llama-4-maverick": {"Total Context": 1050000, "Max Output": 16400},
            # "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": {"Total Context": 131000, "Max Output": 131000}, #已弃用  
            "meta-llama/llama-3.3-70b-instruct:free": {"Total Context": 65536, "Max Output": 65536}, 
            "meta-llama/llama-3.2-3b-instruct:free": {"Total Context": 131000, "Max Output": 131000},  
            "moonshotai/kimi-dev-72b:free": {"Total Context": 131000, "Max Output": 131000}, # 推理 
            "moonshotai/kimi-k2:free": {"Total Context": 32800, "Max Output": 32800}, # 推理 
            "tngtech/deepseek-r1t2-chimera:free": {"Total Context": 163800, "Max Output": 163800}, # 强大的推理性能 
            "deepseek/deepseek-r1-0528:free": {"Total Context": 131100, "Max Output": 131100}, # 推理能力接近 O3 
            "qwen/qwen-turbo": {"Total Context": 1000000, "Max Output": 8200}, 
            "qwen/qwen3-235b-a22b-2507": {"Total Context": 262100, "Max Output": 262100}, # 推理模式 
            "qwen/qwen3-coder:free": {"Total Context": 262100, "Max Output": 262100}, # 长上下文推理 
            "qwen/qwen3-235b-a22b:free": {"Total Context": 131000, "Max Output": 131000}, # 推理模式
            "mistralai/mistral-medium-3.1": {"Total Context": 262100, "Max Output": 262100}, # 最先进的推理能力 
            "x-ai/grok-4": {"Total Context": 256000, "Max Output": 256000},  # xai最新的推理模型 Note that reasoning is not exposed, reasoning cannot be disabled, and the reasoning effort cannot be specified.
            "deepseek/deepseek-chat-v3.1": {"Total Context": 128000, "Max Output": 128000},  #supports both thinking and non-thinking modes via prompt templates.
            "deepseek/deepseek-chat-v3-0324:free": {"Total Context": 163840, "Max Output": 163840}  #supports both thinking and non-thinking modes via prompt templates.
        }
        
        # # 模型成本配置（每1000个token的价格，单位：美元）
        # self.model_costs = {
        #     "openai/gpt-5-mini": {"input": 0.00025, "output": 0.002},
        #     "openai/gpt-4-turbo": {"input": 0.01, "output": 0.03},
        #     "openai/gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        #     "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
        #     "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        #     "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        #     "google/gemini-pro-1.5": {"input": 0.0035, "output": 0.0105},
        #     "google/gemini-flash-1.5": {"input": 0.00075, "output": 0.003},
        #     "meta-llama/llama-3.1-405b-instruct": {"input": 0.005, "output": 0.005},
        #     "meta-llama/llama-3.1-70b-instruct": {"input": 0.0009, "output": 0.0009},
        #     "meta-llama/llama-3.1-8b-instruct": {"input": 0.00018, "output": 0.00018},
        #     "moonshotai/kimi-k2:free": {"input": 0.0, "output": 0.0},  # 免费模型
        #     "deepseek/deepseek-chat": {"input": 0.00014, "output": 0.00028},
        #     "qwen/qwen-2.5-72b-instruct": {"input": 0.0009, "output": 0.0009},
        #     "google/gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
        # }


        # 模型成本配置(每1000个token的价格，单位:美元)
        self.model_costs = {
            "openai/gpt-5-mini": {"input": 0.00025, "output": 0.002},
            "openai/gpt-5": {"input": 0.00125, "output": 0.01},
            "openai/gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
            "anthropic/claude-opus-4.1": {"input": 0.015, "output": 0.075},
            "anthropic/claude-sonnet-4": {"input": 0.003, "output": 0.015},
            "google/gemini-pro-1.5": {"input": 0.00250, "output": 0.01},#>128k.≤128 {"input": 0.00125, "output": 0.005}
            "google/gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
            "google/gemini-2.5-pro-preview": {"input": 0.00250, "output": 0.015},#>200k.≤200 {"input": 0.00125, "output": 0.01}
            "google/gemini-2.5-pro-exp-03-25": {"input": 0.0, "output": 0.0},
            "google/gemini-2.0-flash-exp:free": {"input": 0.0, "output": 0.0},
            "meta-llama/llama-4-scout": {"input": 0.00008, "output": 0.0003},
            "meta-llama/llama-4-maverick": {"input": 0.00015, "output": 0.00085},
            "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": {"input": 0.0, "output": 0.0},
            "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},
            "meta-llama/llama-3.2-3b-instruct:free": {"input": 0.0, "output": 0.0},
            "moonshotai/kimi-dev-72b:free": {"input": 0.0, "output": 0.0},
            "moonshotai/kimi-k2:free": {"input": 0.0, "output": 0.0},
            "tngtech/deepseek-r1t2-chimera:free": {"input": 0.0, "output": 0.0},
            "deepseek/deepseek-r1-0528:free": {"input": 0.0, "output": 0.0},
            "qwen/qwen-turbo": {"input": 0.00005, "output": 0.0002},
            "qwen/qwen3-235b-a22b-2507": {"input": 0.00013, "output": 0.0006},
            "qwen/qwen3-coder:free": {"input": 0.0, "output": 0.0},
            "qwen/qwen3-235b-a22b:free": {"input": 0.0, "output": 0.0},
            "mistralai/mistral-medium-3.1": {"input": 0.0004, "output": 0.002},
            "x-ai/grok-4": {"input": 0.006, "output": 0.03},    #＞128k.≤128k{"input": 0.003, "output": 0.015}
            "deepseek/deepseek-chat-v3.1": {"input": 0.00032, "output": 0.00115},  #DeepSeek-V3.1 is a large hybrid reasoning model (671B parameters, 37B active) that supports both thinking and non-thinking modes via prompt templates. It extends the DeepSeek-V3 base with a two-phase long-context training process, reaching up to 128K tokens, and uses FP8 microscaling for efficient inference.
        }        
    def read_file(self, file_path: str) -> str:
        """读取文件（支持txt和csv格式）"""
        try:
            file_extension = Path(file_path).suffix.lower()

            if file_extension == '.csv':
                return self._read_csv_file(file_path)
            else:
                return self._read_text_file(file_path)

        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {file_path}")
        except Exception as e:
            raise Exception(f"读取文件失败: {str(e)}")

    def _read_text_file(self, file_path: str) -> str:
        """读取纯文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"成功读取文本文件: {file_path}, 长度: {len(content)} 字符")
        return content

    def _read_csv_file(self, file_path: str) -> Tuple[str, int]:
        """读取CSV文件并转换为文本，基于模型上下文长度进行确定性采样"""
        # 获取当前模型的上下文限制
        model_name = self.llm_client.model
        max_context =max( self.model_context_limits.get(model_name)["Total Context"], 32768) # 默认32k
        # 使用上下文长度的75%作为安全边界
        safe_token_limit = int(max_context * 0.75)

        # 模型上下文配置

        content_parts = []
        all_rows = []

        # Try different encodings and delimiters
        encodings = ['utf-8', 'gbk', 'latin-1']
        delimiters = [',', ';', '\t', '|']

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        reader = csv.DictReader(f, delimiter=delimiter)

                        # Try reading first row to validate format
                        first_row = next(reader)
                        fieldnames = reader.fieldnames

                        # If successfully read and has required columns (id, title, text are mandatory)
                        if fieldnames and 'id' in fieldnames and 'text' in fieldnames and 'title' in fieldnames:
                            print(f"Successfully identified CSV format: encoding={encoding}, delimiter='{delimiter}'")
                            # print(f"CSV columns: {fieldnames}")

                            # Reopen file to read all data
                            f.seek(0)
                            reader = csv.DictReader(f, delimiter=delimiter)

                            # 首先读取所有行到内存
                            print("正在读取CSV数据...")
                            for row in reader:
                                if row['id'] and row['text'] and row['title']:
                                    all_rows.append(row)

                            total_rows = len(all_rows)
                            print(f"总共读取到 {total_rows} 行有效数据")
                            
                            # 确定性采样：使用固定种子确保结果可重现
                            random.seed(42)  # 固定种子保证确定性
                            
                            # 计算每行的平均token数来估算需要采样多少行
                            if total_rows > 0:
                                # 先采样少量行来估算平均token数
                                sample_size = min(100, total_rows)
                                sample_rows = random.sample(all_rows, sample_size)

                                total_sample_tokens = 0
                                print("正在估算token数量...")
                                for row in sample_rows:
                                    # 构建包含所有可用信息的文本
                                    row_parts = [f"ID: {row['id']}", f"Title: {row['title']}", f"Text: {row['text']}"]

                                    # 添加其他可选字段
                                    if 'topic' in row and row['topic']:
                                        row_parts.append(f"Topic: {row['topic']}")
                                    if 'Date' in row and row['Date']:
                                        row_parts.append(f"Date: {row['Date']}")
                                    if 'keywords' in row and row['keywords']:
                                        row_parts.append(f"Keywords: {row['keywords']}")

                                    row_text = " | ".join(row_parts)
                                    total_sample_tokens += len(row_text) // self.avg_chars_per_token
                                
                                avg_tokens_per_row = total_sample_tokens / sample_size
                                print(f"平均每行token数: {avg_tokens_per_row:.2f}")
                                
                                # 计算可以包含的最大行数
                                max_rows = int(safe_token_limit / avg_tokens_per_row)
                                max_rows = min(max_rows, total_rows)
                                
                                print(f"基于token限制，最多可包含 {max_rows} 行")
                                
                                # 重新设置种子进行最终采样
                                random.seed(42)
                                if max_rows < total_rows:
                                    selected_rows = random.sample(all_rows, max_rows)
                                    print(f"从 {total_rows} 行中确定性采样了 {max_rows} 行")
                                else:
                                    selected_rows = all_rows
                                    print(f"使用全部 {total_rows} 行数据")
                                
                                # 按原始顺序排序以保持一致性
                                selected_rows.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else 0)
                                
                                # 构建最终内容并验证token数量
                                token_count = 0
                                final_rows = []
                                content_parts = []
                                
                                print("正在构建最终数据...")
                                for row in selected_rows:
                                    # 构建包含所有可用信息的文本
                                    row_parts = [f"ID: {row['id']}", f"Title: {row['title']}", f"Text: {row['text']}"]

                                    # 添加其他可选字段
                                    if 'topic' in row and row['topic']:
                                        row_parts.append(f"Topic: {row['topic']}")
                                    if 'Date' in row and row['Date']:
                                        row_parts.append(f"Date: {row['Date']}")
                                    if 'keywords' in row and row['keywords']:
                                        row_parts.append(f"Keywords: {row['keywords']}")

                                    row_text = " | ".join(row_parts)
                                    new_token_count = token_count + (len(row_text) // self.avg_chars_per_token)

                                    # 如果添加这行会超过限制，则停止
                                    if new_token_count > safe_token_limit:
                                        print(f"警告: 达到token限制，已减少采样行数至 {len(final_rows)} 行")
                                        break

                                    final_rows.append(row)
                                    content_parts.append(row_text)
                                    token_count = new_token_count
                                
                                content = "\n".join(content_parts)
                                print(f"Successfully read CSV file: {file_path}")
                                print(f"Selected rows: {len(final_rows)}, Approximate tokens: {token_count}")
                                print(f"Token usage: {token_count}/{safe_token_limit} ({token_count/safe_token_limit*100:.1f}%)")
                                return content, len(final_rows)
                            else:
                                raise Exception("没有找到有效的数据行")

                except Exception as e:
                    continue  # Try next combination    
        # 如果所有尝试都失败，抛出错误
        raise Exception(f"无法解析CSV文件，请检查文件格式。尝试过的编码: {encodings}, 分隔符: {delimiters}")

    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        return math.ceil(len(text) / self.avg_chars_per_token)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> Tuple[float, str]:
        """估算API调用成本"""
        if model not in self.model_costs:
            return 0.0, f"未知模型 {model} 的成本信息"

        costs = self.model_costs[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost

        cost_info = f"输入: ${input_cost:.6f} ({input_tokens} tokens) + 输出: ${output_cost:.6f} ({output_tokens} tokens) = 总计: ${total_cost:.6f}"
        return total_cost, cost_info

    def create_analysis_prompt(self, text_content: str,row_count: int) -> str:
        """创建分析提示词"""
        # 根据模型上下文限制动态调整文本长度
        model_name = self.llm_client.model
        max_context_len = max( self.model_context_limits.get(model_name)["Total Context"], 32768) * self.avg_chars_per_token
        # max_chars = int(max_context * 0.6 * self.avg_chars_per_token)  # 为prompt留出40%空间
        
        if len(text_content) > max_context_len:
            text_content = text_content[:max_context_len] + "..."
            print(f"文本过长，已截取前{max_context_len}字符进行分析（基于模型{model_name}的上下文限制）")

        # 基于token数量的智能主题数量计算
        estimated_tokens = self.estimate_tokens(text_content)

        # 根据内容复杂度动态调整主题数量
        # 基础逻辑：每1000-1500个token生成1个主题
        base_topics_per_1k_tokens = 0.8  # 每1000个token约0.8个主题

        # 根据数据行数调整密度：行数多说明内容多样性高
        if row_count > 1000:
            density_factor = 1.2  # 大数据集，主题密度更高
        elif row_count > 500:
            density_factor = 1.0  # 中等数据集，标准密度
        else:
            density_factor = 0.8  # 小数据集，主题密度较低

        # 计算基础主题数
        token_based_estimate = int((estimated_tokens / 1000) * base_topics_per_1k_tokens * density_factor)

        # 结合行数进行微调，避免极端情况
        row_based_estimate = row_count // 8  # 每8行约1个主题作为参考

        # 取两者的加权平均，token权重更高
        topic_estimate = int(token_based_estimate * 0.7 + row_based_estimate * 0.3)

        # 移除上限限制，只保留最低限制防止异常
        topic_estimate = max(topic_estimate, 3)

        print(f"智能主题数量计算:")
        print(f"  数据行数: {row_count}, 估算tokens: {estimated_tokens:,}")
        print(f"  Token密度因子: {density_factor}, 基于token: {token_based_estimate}, 基于行数: {row_based_estimate}")
        print(f"  最终主题数: {topic_estimate}")

        # 针对不同模型使用不同的提示词格式
        model_name = self.llm_client.model

        if "qwen" in model_name.lower():
            # Qwen模型专用提示词 - 明确禁止思考模式，要求生成更多主题
            prompt = rf"""You are a professional text analyst. Analyze the provided news data and generate AS MANY distinct topics as possible, aiming for at least {topic_estimate} topics but preferably MORE.

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON format, no additional text or explanations
- Do NOT use <think> tags or reasoning steps
- Do NOT include any thinking process or analysis
- Start directly with [ and end with ]
- Generate AT LEAST {topic_estimate} topics, but aim for MORE if the content supports it
- Be comprehensive and exhaustive in your topic identification

REQUIRED JSON FORMAT:
[
    {{
        "Topic 1": {{
            "Summary": "A concise summary of the topic",
            "Keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
            "Source Titles": ["Exact Title From Input", "Another Exact Title", "Third Title"]
        }}
    }},
    {{
        "Topic 2": {{
            "Summary": "A concise summary of the topic",
            "Keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
            "Source Titles": ["Real Title From Data", "Another Real Title"]
        }}
    }}
]

Text data to analyze:
{text_content}"""
        else:
            # 其他模型使用原始提示词格式
            prompt = rf"""Please conduct a thematic analysis of the provided text data, generating several independent topics. Each topic should balance both generalization and granularity: it must provide an overall description while avoiding being overly vague.

IMPORTANT: For "Source Titles", you MUST copy exact titles from the input data. Look for lines containing "Title: [actual title]" and copy only those exact titles.

Output format:
[
    {{
        "Topic 1": {{
            "Summary": "A one-sentence concise summary capturing the core essence of the topic",
            "Keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5"],
            "Source Titles": ["Exact Title From Input Data", "Another Exact Title From Input", "Third Real Title"]
        }}
    }},
    {{
        "Topic 2": {{
            "Summary": "A one-sentence concise summary capturing the core essence of the topic",
            "Keywords": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5", "Keyword6", "Keyword7"],
            "Source Titles": ["Real Title From Data", "Another Real Title"]
        }}
    }}
],
……

Specific requirements:

Generate AS MANY topics as possible, aiming for at least {max(topic_estimate, 3)} topics but preferably MORE. Be comprehensive and exhaustive in your analysis. The first key in the dictionary (e.g., "Topic 1") serves as a numerical index for the topic.

FLEXIBLE REQUIREMENTS (QUALITY OVER QUANTITY):
- Each topic should include 5-12 keywords that are truly relevant and meaningful
- Each topic should include 3-8 source titles that actually belong to this topic
- DO NOT pad with irrelevant keywords or titles just to meet a number
- Focus on accuracy and relevance rather than hitting exact counts
- If a topic naturally has fewer keywords/titles, that's perfectly fine

Keywords Guidelines:
- Include 5-12 keywords per topic (flexible based on actual relevance)
- Keywords should be genuinely representative of the topic
- Avoid generic or filler keywords
- Each keyword should add meaningful information about the topic

Source Titles Guidelines:
- Include 3-8 titles per topic that genuinely belong to this topic
- **MANDATORY: Use ONLY exact titles from the input data (copy them exactly as they appear after "Title:")**
- **DO NOT create, modify, or invent any titles - they must exist in the input data**
- Select titles that best represent the topic's core themes
- DO NOT use generic placeholders like "Title1", "Title2"
- If fewer titles truly belong to a topic, include fewer titles
- Quality and relevance matter more than quantity
- **CRITICAL: Within each topic, do NOT repeat the same title multiple times**
- **Each title should appear only once within the same topic's Source Titles array**
- **VERIFICATION: Every title you include must be findable in the input text after "Title:"**

Semantic relevance: Keywords and titles within the same topic must share clear semantic relatedness, ensuring internal consistency.

Avoid redundancy:
- Minimize repeated keywords across topics to maintain distinction and reduce overlap
- **Within each topic, never repeat the same source title**
- **Each source title should appear only once per topic**

CRITICAL JSON FORMAT REQUIREMENTS:
- Output ONLY valid JSON format, no additional text before or after
- Start with [ and end with ]
- Use double quotes for all strings
- Avoid special characters in strings (use simple text only)
- Keep summaries and keywords concise to avoid JSON parsing issues
- Ensure the JSON is complete and properly closed
- If response is getting too long, prioritize fewer topics with complete information

Text contents to analyze:
{text_content}
"""

        return prompt

    def extract_real_titles_from_data(self, text_content: str) -> tuple:
        """从输入数据中提取所有真实的标题及其对应的ID"""
        real_titles = set()
        title_to_id = {}  # 标题到ID的映射
        lines = text_content.split('\n')

        for line in lines:
            # 查找 "ID: " 和 "Title: "
            if 'ID: ' in line and 'Title: ' in line:
                # 提取ID
                id_start = line.find('ID: ') + 4
                id_end = line.find(' | ', id_start)
                if id_end == -1:
                    continue
                data_id = line[id_start:id_end].strip()

                # 提取Title
                title_start = line.find('Title: ') + 7
                title_end = line.find(' | ', title_start)
                if title_end == -1:
                    title_end = len(line)

                title = line[title_start:title_end].strip()
                if title and data_id:
                    real_titles.add(title)
                    title_to_id[title] = data_id

        # 静默提取，不输出日志
        return real_titles, title_to_id

    def load_full_dataset(self, file_path: str) -> List[Dict]:
        """加载完整数据集用于关键词匹配"""
        all_data = []

        encodings = ['utf-8', 'gbk', 'latin-1']
        delimiters = [',', ';', '\t', '|']

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        reader = csv.DictReader(f, delimiter=delimiter)

                        # 验证是否有必需的列
                        first_row = next(reader)
                        fieldnames = reader.fieldnames

                        if fieldnames and 'id' in fieldnames and 'text' in fieldnames and 'title' in fieldnames:
                            print(f"加载完整数据集: encoding={encoding}, delimiter='{delimiter}'")

                            # 重新开始读取
                            f.seek(0)
                            reader = csv.DictReader(f, delimiter=delimiter)

                            for row in reader:
                                if row.get('id') and row.get('title') and row.get('text'):
                                    all_data.append({
                                        'id': row['id'],
                                        'title': row['title'].strip(),
                                        'text': row['text'].strip(),
                                        'topic': row.get('topic', '').strip(),
                                        'keywords': row.get('keywords', '').strip()
                                    })

                            print(f"成功加载完整数据集: {len(all_data):,} 行")
                            return all_data

                except Exception as e:
                    continue

        print("警告: 无法加载完整数据集，将使用采样数据进行标题验证")
        return []

    def keyword_based_title_matching(self, keywords: List[str], dataset: List[Dict], max_matches: int = None) -> List[Dict]:
        """基于关键词在完整数据集中匹配标题"""
        if not dataset or not keywords:
            return []

        matches = []

        # 创建关键词的正则表达式（不区分大小写）
        keyword_patterns = []
        for keyword in keywords:
            if len(keyword.strip()) > 1:  # 过滤太短的关键词
                escaped_keyword = re.escape(keyword.strip())
                pattern = rf'\b{escaped_keyword}\b'
                keyword_patterns.append((keyword, re.compile(pattern, re.IGNORECASE)))

        for item in dataset:
            title = item['title']
            text = item['text']
            combined_text = f"{title} {text}"

            # 计算匹配的关键词
            matched_keywords = []
            match_score = 0

            for keyword, pattern in keyword_patterns:
                if pattern.search(combined_text):
                    matched_keywords.append(keyword)
                    # 标题中的匹配权重更高
                    if pattern.search(title):
                        match_score += 2
                    else:
                        match_score += 1

            if matched_keywords:
                matches.append({
                    'id': item['id'],
                    'title': item['title'],
                    'matched_keywords': matched_keywords,
                    'match_score': match_score,
                    'match_count': len(matched_keywords)
                })

        # 按匹配分数排序，去重
        seen_titles = set()
        unique_matches = []
        for match in sorted(matches, key=lambda x: (x['match_score'], x['match_count']), reverse=True):
            if match['title'] not in seen_titles:
                unique_matches.append(match)
                seen_titles.add(match['title'])

        # 如果没有指定最大匹配数，返回所有匹配；否则限制数量
        if max_matches is None:
            return unique_matches
        else:
            return unique_matches[:max_matches]

    def enhance_topics_with_keyword_matching(self, topics: List[Dict[str, any]], file_path: str) -> List[Dict[str, any]]:
        """使用关键词匹配增强主题的标题覆盖"""
        # 加载完整数据集
        full_dataset = self.load_full_dataset(file_path)

        if not full_dataset:
            return topics

        enhanced_topics = []

        for topic in topics:
            topic_num = topic.get('topic_num', 0)
            keywords = topic.get('keywords', [])

            # 使用自然的关键词匹配，不设固定数量限制
            # 让每个主题根据实际匹配情况获得不同数量的文本
            keyword_matches = self.keyword_based_title_matching(keywords, full_dataset, max_matches=None)

            # 根据匹配质量和关键词特性进行自然筛选
            if keyword_matches:
                # 方法1: 基于匹配分数的自适应筛选
                scores = [match['match_score'] for match in keyword_matches]
                if len(scores) > 1:
                    # 动态阈值：根据分数分布决定保留比例
                    import numpy as np
                    score_mean = np.mean(scores)
                    score_std = np.std(scores)

                    # 如果分数差异大，使用较高阈值；如果分数相近，使用较低阈值
                    if score_std > score_mean * 0.3:  # 分数差异较大
                        min_score = score_mean + score_std * 0.5
                    else:  # 分数相近，更宽松的筛选
                        min_score = score_mean

                    keyword_matches = [m for m in keyword_matches if m['match_score'] >= min_score]

                # 方法2: 基于关键词特异性的动态限制
                keyword_specificity = len(set(keywords))  # 关键词的多样性
                if keyword_specificity <= 3:  # 通用关键词，限制更严格
                    max_allowed = min(15, len(keyword_matches))
                elif keyword_specificity <= 6:  # 中等特异性
                    max_allowed = min(25, len(keyword_matches))
                else:  # 高特异性关键词，允许更多匹配
                    max_allowed = min(40, len(keyword_matches))

                keyword_matches = keyword_matches[:max_allowed]

            # 转换为带ID的格式
            enhanced_titles = []
            for match in keyword_matches:
                enhanced_titles.append({
                    'title': match['title'],
                    'id': match['id'],
                    'match_score': match['match_score'],
                    'matched_keywords': match['matched_keywords']
                })

            # 创建增强后的主题，只保留必要字段
            enhanced_topic = {
                'topic_num': topic.get('topic_num', topic_num),
                'summary': topic.get('summary', ''),
                'keywords': topic.get('keywords', []),
                'keyword_count': topic.get('keyword_count', len(keywords)),
                'meets_requirements': topic.get('meets_requirements', True),
                'source_titles_with_ids': enhanced_titles,
                'enhancement_info': {
                    'method': 'keyword_matching',
                    'full_dataset_size': len(full_dataset),
                    'matches_found': len(keyword_matches)
                }
            }

            enhanced_topics.append(enhanced_topic)

        # 关键词匹配增强完成
        return enhanced_topics

    def validate_and_fix_titles(self, topics: List[Dict[str, any]], real_titles: set, title_to_id: dict) -> List[Dict[str, any]]:
        """验证并修复主题中的标题，确保都是真实存在的"""
        if not topics or not real_titles:
            return topics

        fixed_topics = []
        total_invalid_titles = 0

        for topic in topics:
            topic_num = topic.get('topic_num', 0)

            # 处理不同的数据结构
            if 'source_titles_with_ids' in topic:
                # 如果已经有ID信息，直接验证
                source_titles_with_ids = topic.get('source_titles_with_ids', [])
                valid_titles_with_ids = []

                for title_info in source_titles_with_ids:
                    title = title_info.get('title', '')
                    if title in real_titles:
                        valid_titles_with_ids.append(title_info)
                    else:
                        total_invalid_titles += 1

                topic_copy = topic.copy()
                topic_copy['source_titles_with_ids'] = valid_titles_with_ids
                fixed_topics.append(topic_copy)

            else:
                # 如果没有source_titles_with_ids，创建空的
                topic_copy = topic.copy()
                topic_copy['source_titles_with_ids'] = []
                fixed_topics.append(topic_copy)

        # 静默处理，不输出验证结果

        return fixed_topics

    def remove_duplicate_titles(self, topics: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """移除每个主题内部重复的source titles"""
        if not topics:
            return topics

        cleaned_topics = []

        for topic in topics:
            # 只处理新的数据结构：source_titles_with_ids
            if 'source_titles_with_ids' in topic:
                source_titles_with_ids = topic.get('source_titles_with_ids', [])

                # 移除同一主题内的重复标题，保持顺序
                seen_titles = set()
                unique_titles_with_ids = []

                for title_info in source_titles_with_ids:
                    title = title_info.get('title', '')
                    if title not in seen_titles:
                        unique_titles_with_ids.append(title_info)
                        seen_titles.add(title)

                # 更新topic的source_titles_with_ids
                topic_copy = topic.copy()
                topic_copy['source_titles_with_ids'] = unique_titles_with_ids
                cleaned_topics.append(topic_copy)
            else:
                # 如果没有source_titles_with_ids，直接复制
                cleaned_topics.append(topic.copy())

        return cleaned_topics

    def parse_topics(self, llm_response: str) -> List[Dict[str, any]]:
        """解析LLM返回的主题结果"""
        topics = []

        try:
            # 针对Qwen模型的特殊处理 - 移除思考标签
            model_name = self.llm_client.model
            cleaned = llm_response.strip()

            if "qwen" in model_name.lower():
                # 移除Qwen模型的思考标签
                if '<think>' in cleaned:
                    # 提取</think>之后的内容
                    think_end = cleaned.find('</think>')
                    if think_end != -1:
                        cleaned = cleaned[think_end + 8:].strip()
                    else:
                        # 如果没有结束标签，尝试找到JSON开始
                        json_start = cleaned.find('[')
                        if json_start != -1:
                            cleaned = cleaned[json_start:].strip()

            # 使用改进的JSON提取方法
            json_str = self._extract_valid_json_portion(cleaned)

            if not json_str:
                print("Warning: 无法找到有效的JSON数组格式")
                return topics
            
            # 清理JSON字符串中的控制字符
            import re
            # 移除或替换常见的控制字符
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # 移除控制字符
            json_str = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')  # 转义换行符等

            # 直接解析JSON格式
            try:
                parsed_data = json.loads(json_str)
                
                if not isinstance(parsed_data, list):
                    print("Warning: 解析结果不是列表格式")
                    return topics
                
                for idx, topic_data in enumerate(parsed_data, 1):
                    if not isinstance(topic_data, dict):
                        continue
                        
                    # 获取第一个键作为主题名
                    topic_name = next(iter(topic_data.keys())) if topic_data else f"Topic {idx}"
                    topic_content = topic_data.get(topic_name, {})
                    
                    if not isinstance(topic_content, dict):
                        continue
                        
                    summary = topic_content.get("Summary", "")
                    keywords = topic_content.get("Keywords", [])
                    source_titles = topic_content.get("Source Titles", [])
                    
                    # 验证关键词数量是否符合要求（更灵活的范围）
                    keyword_count = len(keywords)
                    meets_requirements = 5 <= keyword_count <= 12
                    
                    topics.append({
                        'topic_num': idx,
                        'summary': summary,
                        'keywords': keywords,
                        'keyword_count': keyword_count,
                        'meets_requirements': meets_requirements
                    })
                    
                    if not meets_requirements:
                        print(f"Warning: Topic {idx} has {keyword_count} keywords (requirement: 5-12)")
                
                # 清理重复的source titles
                topics = self.remove_duplicate_titles(topics)
                return topics
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                print(f"错误位置: 第{e.lineno}行, 第{e.colno}列")

                # 保存原始响应用于调试
                debug_file = f"debug_json_error_{int(time.time())}.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"JSON解析错误: {str(e)}\n")
                    f.write(f"错误位置: 第{e.lineno}行, 第{e.colno}列\n\n")
                    f.write("原始响应:\n")
                    f.write(llm_response)
                    f.write("\n\n提取的JSON部分:\n")
                    f.write(json_str)
                print(f"调试信息已保存到: {debug_file}")

                print(f"尝试修复JSON格式...")

                # 尝试修复常见的JSON格式问题
                try:
                    # 方法1: 尝试修复引号问题
                    fixed_json = self._fix_json_format(json_str)
                    parsed_data = json.loads(fixed_json)
                    print("JSON修复成功！")
                except:
                    # 方法2: 尝试使用正则表达式提取主题信息
                    print("JSON修复失败，尝试正则表达式解析...")
                    topics = self._parse_topics_with_regex(llm_response)
                    if topics:
                        print(f"正则表达式解析成功，提取到 {len(topics)} 个主题")
                        # 清理重复的source titles
                        topics = self.remove_duplicate_titles(topics)
                        return topics
                    else:
                        # 方法3: 尝试紧急解析
                        print("正则表达式解析失败，尝试紧急解析...")
                        topics = self._emergency_parse(llm_response)
                        if topics:
                            print(f"紧急解析成功，提取到 {len(topics)} 个主题")
                            topics = self.remove_duplicate_titles(topics)
                            return topics
                        else:
                            print("所有解析方法都失败了")
                            return []
                
        except Exception as e:
            print(f"Error parsing topics: {str(e)}")
            print(f"LLM Response preview: {llm_response[:500]}...")
            return []

    def _fix_json_format(self, json_str: str) -> str:
        """尝试修复常见的JSON格式问题"""
        import re

        # 修复常见问题
        fixed = json_str

        # 1. 移除可能的BOM和控制字符
        fixed = fixed.replace('\ufeff', '')  # BOM
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', fixed)

        # 2. 修复多余的逗号
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)

        # 3. 修复缺失的逗号
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)

        # 4. 修复未闭合的字符串（简单情况）
        fixed = re.sub(r'"([^"]*)\n([^"]*)"', r'"\1 \2"', fixed)

        # 5. 修复换行符在字符串中的问题
        fixed = re.sub(r'"\s*\n\s*"', '""', fixed)

        # 6. 尝试修复截断的JSON（如果以逗号结尾）
        fixed = fixed.rstrip()
        if fixed.endswith(','):
            fixed = fixed[:-1]

        # 7. 确保JSON以]结尾
        if not fixed.endswith(']') and not fixed.endswith('}'):
            # 尝试找到最后一个完整的对象
            last_brace = fixed.rfind('}')
            if last_brace != -1:
                fixed = fixed[:last_brace + 1] + ']'

        return fixed

    def _extract_valid_json_portion(self, response: str) -> str:
        """提取响应中可能有效的JSON部分"""
        # 首先尝试移除markdown代码块标记
        cleaned_response = response

        # 移除 ```json 和 ``` 标记
        import re
        # 查找并移除markdown代码块
        json_block_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_block_pattern, response, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(1).strip()
            print("检测到markdown代码块，已提取JSON内容")

        # 查找第一个[和最后一个]
        start_idx = cleaned_response.find('[')
        if start_idx == -1:
            return ""

        # 从后往前找[最后一个]
        end_idx = -1
        bracket_count = 0
        for i in range(len(cleaned_response) - 1, start_idx - 1, -1):
            if cleaned_response[i] == ']':
                bracket_count += 1
                if bracket_count == 1:
                    end_idx = i
                    break
            elif cleaned_response[i] == '[':
                bracket_count -= 1

        if end_idx == -1:
            # 如果没找到匹配的]，尝试找到最后一个}然后添加]
            last_brace = cleaned_response.rfind('}')
            if last_brace > start_idx:
                return cleaned_response[start_idx:last_brace + 1] + ']'
            return ""

        return cleaned_response[start_idx:end_idx + 1]

    def _parse_topics_with_regex(self, response: str) -> List[Dict[str, any]]:
        """使用正则表达式解析主题（备用方法）"""
        import re
        topics = []

        try:
            print("开始正则表达式解析...")

            # 清理特殊字符和编码问题
            print("清理编码问题...")
            cleaned_response = response.replace('â', '"').replace('â', '"').replace('â', '-')
            cleaned_response = cleaned_response.replace('\x80\x98', '"').replace('\x80\x99', '"').replace('\x80?', '"')
            cleaned_response = cleaned_response.replace('\\x80\\x98', '"').replace('\\x80\\x99', '"').replace('\\x80?', '"')

            # 方法1: 尝试找到完整的主题块
            print("查找主题块...")
            topic_pattern = r'"Topic\s+(\d+)":\s*\{[^}]*"Summary":\s*"([^"]*)"[^}]*"Keywords":\s*\[([^\]]*)\][^}]*"Source Titles":\s*\[([^\]]*)\]'
            matches = re.findall(topic_pattern, cleaned_response, re.DOTALL)

            print(f"找到 {len(matches)} 个完整主题块")

            for match in matches:
                topic_num, summary, keywords_str, titles_str = match

                # 解析关键词
                keywords = []
                if keywords_str:
                    keywords = [k.strip().strip('"') for k in keywords_str.split(',') if k.strip()]

                # 解析标题（限制为5个）
                source_titles = []
                if titles_str:
                    # 处理可能被截断的标题列表
                    titles_raw = titles_str.split('", "')
                    for title in titles_raw:
                        title = title.strip().strip('"')
                        if title and len(title) > 3:  # 过滤掉太短的片段
                            source_titles.append(title)
                            if len(source_titles) >= 5:  # 限制为5个titles
                                break

                if summary and keywords:  # 只有当有summary和keywords时才添加
                    topics.append({
                        'topic_num': int(topic_num),
                        'summary': summary,
                        'keywords': keywords,
                        'keyword_count': len(keywords),
                        'meets_requirements': 5 <= len(keywords) <= 12
                    })

            # 方法2: 如果方法1失败，尝试更宽松的匹配
            if not topics:
                print("尝试宽松匹配...")
                # 查找Summary
                summary_matches = re.findall(r'"Summary":\s*"([^"]*)"', cleaned_response)
                # 查找Keywords
                keywords_matches = re.findall(r'"Keywords":\s*\[([^\]]*)\]', cleaned_response)
                # 查找Source Titles
                titles_matches = re.findall(r'"Source Titles":\s*\[([^\]]*)\]', cleaned_response)

                print(f"找到 {len(summary_matches)} 个摘要, {len(keywords_matches)} 个关键词组, {len(titles_matches)} 个标题组")

                # 组合结果
                max_topics = min(len(summary_matches), len(keywords_matches))
                for i in range(max_topics):
                    summary = summary_matches[i]
                    keywords_str = keywords_matches[i]
                    titles_str = titles_matches[i] if i < len(titles_matches) else ""

                    keywords = [k.strip().strip('"') for k in keywords_str.split(',') if k.strip()]
                    source_titles = []
                    if titles_str:
                        titles_raw = titles_str.split('", "')
                        for title in titles_raw:
                            title = title.strip().strip('"')
                            if title and len(title) > 3:
                                source_titles.append(title)
                                if len(source_titles) >= 5:  # 限制为5个titles
                                    break

                    topics.append({
                        'topic_num': i + 1,
                        'summary': summary,
                        'keywords': keywords,
                        'keyword_count': len(keywords),
                        'meets_requirements': 5 <= len(keywords) <= 12
                    })

            print(f"正则表达式解析完成，提取到 {len(topics)} 个主题")
            # 清理重复的source titles
            topics = self.remove_duplicate_titles(topics)
            return topics

        except Exception as e:
            print(f"正则表达式解析也失败了: {str(e)}")
            return []

    def _emergency_parse(self, response: str) -> List[Dict[str, any]]:
        """紧急解析方法 - 尽可能提取任何有用信息"""
        topics = []

        try:
            print("开始紧急解析...")

            # 清理响应
            cleaned = response.replace('\n', ' ').replace('\r', ' ')

            # 尝试找到任何Summary信息
            import re
            summaries = re.findall(r'"Summary":\s*"([^"]{10,200})"', cleaned)
            keywords_groups = re.findall(r'"Keywords":\s*\[([^\]]{10,500})\]', cleaned)
            titles_groups = re.findall(r'"Source Titles":\s*\[([^\]]{10,1000})\]', cleaned)

            print(f"找到 {len(summaries)} 个摘要, {len(keywords_groups)} 个关键词组, {len(titles_groups)} 个标题组")

            # 创建主题
            for i in range(min(len(summaries), len(keywords_groups))):
                summary = summaries[i]
                keywords_str = keywords_groups[i]
                titles_str = titles_groups[i] if i < len(titles_groups) else ""

                # 解析关键词
                keywords = []
                for kw in keywords_str.split(','):
                    kw = kw.strip().strip('"').strip("'")
                    if kw and len(kw) > 1:
                        keywords.append(kw)

                # 解析标题（不超过5个）
                source_titles = []
                if titles_str:
                    for title in titles_str.split('", "'):
                        title = title.strip().strip('"').strip("'")
                        if title and len(title) > 3:
                            source_titles.append(title)
                            if len(source_titles) >= 5:  # 限制为5个titles
                                break

                if summary and keywords:
                    final_keywords = keywords[:12]  # 限制关键词数量
                    keyword_count = len(final_keywords)
                    topics.append({
                        'topic_num': i + 1,
                        'summary': summary,
                        'keywords': final_keywords,
                        'keyword_count': keyword_count,
                        'meets_requirements': 5 <= keyword_count <= 12,
                        'emergency_parsed': True
                    })

            print(f"紧急解析完成，提取到 {len(topics)} 个主题")
            # 清理重复的source titles
            topics = self.remove_duplicate_titles(topics)
            return topics

        except Exception as e:
            print(f"紧急解析失败: {e}")
            return []
    
    # def parse_topics(self, llm_response: str) -> List[Dict[str, any]]:
    #     """解析LLM返回的主题结果"""
    #     topics = []
        
    #     # Find the first [ and last ] in the response, including them
    #     content_match = re.match(r'.*?(\[.*\]).*', llm_response, re.DOTALL)
    #     if not content_match:
    #         return topics
    #     # Convert string list to actual list by evaluating it
    #     cleaned = content_match.group(1).strip()
    #     try:
    #         # Safely evaluate the string as a Python expression
    #         theme_list = eval(cleaned)
    #         if not isinstance(theme_list, list):
    #             raise ValueError("Parsed content is not a list")    
                
    #     except:
    #         # Fallback to original string if evaluation fails
    #         pass
        

            
    #     for i, theme_dict in enumerate(theme_list):
    #         # Each theme_dict should have format:
    #         # {"Theme N": {"Summary": "...", "Keywords": [...], "Source Line Indices": [...]}}
            
    #         theme_data = list(theme_dict.values())[0]  # Get the inner dict
            
    #         summary = theme_data.get("Summary", "").strip()
    #         keywords = [k.strip() for k in theme_data.get("Keywords", [])]
    #         indices = [int(idx) for idx in theme_data.get("Source Line Indices", [])]
            
    #         keyword_count = len(keywords)
    #         meets_requirements = 5 <= keyword_count <= 8
            
    #         topics.append({
    #             'topic_num': i,
    #             'summary': summary,
    #             'keywords': keywords,
    #             'source_indices': indices,
    #             'keyword_count': keyword_count,
    #             'meets_requirements': meets_requirements
    #         })
            
    #         if not meets_requirements:
    #             print(f"Warning: Theme {i} has {keyword_count} keywords (requirement: 5-8)")
        
        # return topics
    
    def analyze_text(self, file_path: str) -> Dict[str, any]:
        """分析文件并生成主题（支持txt和csv格式）"""
        print(f"开始分析文件: {file_path}")

        # 读取文件（自动识别格式）
        text_content, row_count = self.read_file(file_path)
        
        # 创建分析提示词
        prompt = self.create_analysis_prompt(text_content,row_count)
        
        # 估算输入token数量
        input_tokens = self.estimate_tokens(prompt)
        print(f"估算输入token数量: {input_tokens}")
        
        # 估算预期输出token数量（基于主题数量的经验估算）
        expected_output_tokens = min(10000, max(1000, row_count * 50))  # 每个主题大约50个token
        
        # 估算成本
        model_name = self.llm_client.model
        estimated_cost, cost_info = self.estimate_cost(input_tokens, expected_output_tokens, model_name)
        print(f"使用模型: {model_name}")
        print(f"预估成本: {cost_info}")
        
        # 重试机制：最多重试3次
        max_retries = 1
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        all_responses = []
        topics = []  # 初始化topics变量
        
        for retry in range(max_retries):
            print(f"正在调用LLM API... (第{retry + 1}次尝试)")
            import time
            start_time = time.time()
            try:
                # 针对不同模型调整参数 - 大幅增加token限制支持更多主题
                model_name = self.llm_client.model
                if "qwen" in model_name.lower():
                    max_tokens = 25000  # Qwen模型大幅增加tokens支持更多主题
                    temperature = 0.0   # Qwen模型使用0温度避免思考模式
                else:
                    max_tokens = 20000  # 其他模型也大幅增加tokens
                    temperature = 0

                llm_response = self.llm_client.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)
                end_time = time.time()
                # API调用完成
                all_responses.append(llm_response)
                break

            except Exception as e:
                if retry == max_retries - 1:
                    raise Exception(f"LLM API调用失败: {str(e)}")
                else:
                    print(f"第{retry + 1}次API调用失败，准备重试: {str(e)}")
                    time.sleep(1)  # 短暂等待后重试
                    continue
            
        # 计算当前输出token数量和成本
        current_output_tokens = self.estimate_tokens(llm_response)
        current_cost, current_cost_info = self.estimate_cost(input_tokens, current_output_tokens, model_name)

        # 累计统计
        total_input_tokens += input_tokens
        total_output_tokens += current_output_tokens
        total_cost += current_cost

        print(f"第1次尝试 - 输出token数量: {current_output_tokens}")
        print(f"第1次尝试 - 成本: {current_cost_info}")

        # 尝试解析结果
        print("开始解析LLM响应...")
        topics = self.parse_topics(llm_response)

        if len(topics) > 0:
            print(f"第1次尝试成功解析出 {len(topics)} 个主题")

            # 使用关键词匹配增强主题标题（新功能）
            print("开始使用关键词匹配增强主题...")
            topics = self.enhance_topics_with_keyword_matching(topics, file_path)

            print(f"总成本: ${total_cost:.6f} (输入tokens: {total_input_tokens}, 输出tokens: {total_output_tokens})")
        else:
            print(f"第1次尝试解析失败，未能提取到有效主题")
        
        # 检查最终是否解析成功
        if len(topics) == 0:
            print(f"警告：经过{max_retries}次重试后，LLM输出格式仍然解析失败")
            print(f"总成本: ${total_cost:.6f} (输入tokens: {total_input_tokens}, 输出tokens: {total_output_tokens})")

            # 尝试最后一次强制解析
            print("尝试最后一次强制解析...")
            try:
                # 保存原始响应到文件以便调试
                debug_file = f"debug_response_{int(time.time())}.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Raw Response:\n{llm_response}\n\n")
                    f.write(f"Response Length: {len(llm_response)}\n")
                print(f"原始响应已保存到: {debug_file}")

                # 尝试强制提取任何可能的信息
                topics = self._emergency_parse(llm_response)
                if topics:
                    print(f"紧急解析成功，提取到 {len(topics)} 个主题")

            except Exception as e:
                print(f"紧急解析也失败了: {e}")

            result = {
                'file_path': file_path,
                'total_topics': len(topics),
                'topics': topics,
                'raw_response': llm_response,
                'all_responses': all_responses,
                'retry_info': {
                    'total_retries': max_retries,
                    'format_parse_failed': len(topics) == 0
                },
                'cost_analysis': {
                    'model': model_name,
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'estimated_cost': estimated_cost,
                    'actual_cost': total_cost,
                    'cost_info': f"总计: ${total_cost:.6f} {'(格式解析失败)' if len(topics) == 0 else '(紧急解析成功)'}"
                },
                'error': f"经过{max_retries}次重试后仍无法解析LLM输出格式" if len(topics) == 0 else None,
                'format_parse_failed': len(topics) == 0,
                'emergency_parsed': len(topics) > 0 and len(topics) != len(self.parse_topics(llm_response))
            }

            if len(topics) == 0:
                print(f"格式解析失败，返回错误结果")
            else:
                print(f"紧急解析成功，返回部分结果")
            return result
        
        result = {
            'file_path': file_path,
            'total_topics': len(topics),
            'topics': topics,
            'raw_response': llm_response,
            'all_responses': all_responses,
            'retry_info': {
                'total_retries': retry + 1,
                'format_parse_failed': False
            },
            'cost_analysis': {
                'model': model_name,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'estimated_cost': estimated_cost,
                'actual_cost': total_cost,
                'cost_info': f"总计: ${total_cost:.6f}"
            }
        }
        
        return result

    def analyze_text_hybrid(self, file_path: str) -> Dict[str, any]:
        """混合方案：LLM生成关键词 + 关键词匹配全数据集"""
        print(f"开始混合方案分析文件: {file_path}")
        print("方案说明: 1) LLM对采样数据生成高质量关键词 2) 关键词匹配全数据集找相关标题")

        # 第一阶段：使用采样数据生成关键词
        print("\n=== 第一阶段：LLM生成关键词 ===")
        text_content, row_count = self.read_file(file_path)

        # 创建专门用于关键词生成的提示词
        prompt = self.create_keyword_generation_prompt(text_content, row_count)

        # 估算输入token数量
        input_tokens = self.estimate_tokens(prompt)
        print(f"估算输入token数量: {input_tokens}")

        # 调用LLM生成关键词
        model_name = self.llm_client.model
        estimated_cost, cost_info = self.estimate_cost(input_tokens, 3000, model_name)
        print(f"使用模型: {model_name}")
        print(f"预估成本: {cost_info}")

        try:
            print("正在调用LLM生成关键词...")
            import time
            start_time = time.time()
            llm_response = self.llm_client.call_llm(prompt, max_tokens=8000, temperature=0)  # 增加关键词生成的token限制
            end_time = time.time()
            print(f"LLM调用完成，耗时: {end_time - start_time:.2f}秒")

            # 解析关键词
            topics = self.parse_topics(llm_response)

            if not topics:
                print("LLM关键词生成失败，回退到标准方案")
                return self.analyze_text(file_path)

            print(f"成功生成 {len(topics)} 个主题的关键词")

        except Exception as e:
            print(f"LLM调用失败: {e}")
            print("回退到标准方案")
            return self.analyze_text(file_path)

        # 第二阶段：关键词匹配全数据集
        print(f"\n=== 第二阶段：关键词匹配全数据集 ===")
        enhanced_topics = self.enhance_topics_with_keyword_matching(topics, file_path)

        # 计算实际成本
        output_tokens = self.estimate_tokens(llm_response)
        actual_cost, actual_cost_info = self.estimate_cost(input_tokens, output_tokens, model_name)

        # 构建结果
        result = {
            'file_path': file_path,
            'method': 'hybrid_keyword_matching',
            'total_topics': len(enhanced_topics),
            'topics': enhanced_topics,
            'raw_response': llm_response,
            'cost_analysis': {
                'model': model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_cost': actual_cost,
                'cost_breakdown': actual_cost_info
            },
            'method_info': {
                'stage1': 'LLM keyword generation from sampled data',
                'stage2': 'Keyword matching across full dataset',
                'advantages': [
                    'Full dataset coverage (100% vs 1%)',
                    'Low cost (keyword matching is free)',
                    'High precision (LLM-generated keywords)',
                    'Scalable to any dataset size'
                ]
            }
        }

        print(f"\n=== 混合方案完成 ===")
        print(f"总成本: ${actual_cost:.6f}")
        print(f"数据覆盖率: 100% (vs 传统方案的 ~1%)")
        print(f"找到主题: {len(enhanced_topics)} 个")

        # 简化统计输出
        total_matches = sum(len(topic.get('source_titles_with_ids', [])) for topic in enhanced_topics)
        print(f"关键词匹配完成: {len(enhanced_topics)}个主题，共{total_matches}个匹配")

        return result

    def create_keyword_generation_prompt(self, text_content: str, row_count: int) -> str:
        """创建专门用于关键词生成的提示词"""
        # 根据模型上下文限制动态调整文本长度
        model_name = self.llm_client.model
        max_context_len = max(self.model_context_limits.get(model_name, {}).get("Total Context", 32768), 32768) * self.avg_chars_per_token

        if len(text_content) > max_context_len:
            text_content = text_content[:max_context_len] + "..."
            print(f"文本过长，已截取前{max_context_len}字符进行分析（基于模型{model_name}的上下文限制）")

        # 基于token数量的智能主题数量计算（关键词生成模式）
        estimated_tokens = self.estimate_tokens(text_content)

        # 关键词生成模式：相对保守，确保质量
        base_topics_per_1k_tokens = 0.6  # 每1000个token约0.6个主题

        # 根据数据行数调整
        if row_count > 800:
            density_factor = 1.1
        elif row_count > 400:
            density_factor = 1.0
        else:
            density_factor = 0.9

        token_based_estimate = int((estimated_tokens / 1000) * base_topics_per_1k_tokens * density_factor)
        row_based_estimate = row_count // 10  # 关键词模式更保守

        # 加权平均
        topic_estimate = int(token_based_estimate * 0.6 + row_based_estimate * 0.4)

        # 移除上限限制，只保留最低限制防止异常
        topic_estimate = max(topic_estimate, 3)

        print(f"关键词生成模式 - 智能主题计算:")
        print(f"  估算tokens: {estimated_tokens:,}, 最终主题数: {topic_estimate}")

        prompt = rf"""Please analyze the provided text data and generate high-quality topics with keywords for comprehensive dataset matching.

TASK: Generate AS MANY distinct topics as possible, aiming for at least {max(topic_estimate, 3)} topics but preferably MORE. Each topic should have 5-12 highly relevant keywords that can be used to find related articles in a large dataset. Be comprehensive and exhaustive.

OUTPUT FORMAT (JSON only):
[
    {{
        "Topic 1": {{
            "Summary": "One-sentence summary of the topic",
            "Keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"]
        }}
    }},
    {{
        "Topic 2": {{
            "Summary": "One-sentence summary of the topic",
            "Keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
        }}
    }}
]

KEYWORD REQUIREMENTS:
- Focus on ENTITIES: proper nouns, names, places, organizations, specific terms
- Include variations: "Bush", "George W Bush", "President Bush"
- Mix broad and specific terms: "terrorism" + "Al Qaeda" + "September 11"
- Avoid generic words like "the", "and", "important", "major"
- Each keyword should be a strong signal for finding related articles
- Keywords will be used for regex matching across 100,000+ articles

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON, no additional text
- Start with [ and end with ]
- Use double quotes for all strings
- Ensure JSON is complete and properly formatted
- Generate AS MANY topics as possible, at least {max(topic_estimate, 3)} topics but preferably MORE

Text data to analyze:
{text_content}
"""

        return prompt

    def save_results(self, results: Dict[str, any], output_file: str = "topic_analysis_results.json"):
        """保存分析结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    

def main():
    """主函数"""
    print("=== 文本主题分析器 (OpenRouter版) - 批量推理模式 ===")

    # ========== 批量推理配置区域 ==========
    # 1. 设置OpenRouter API密钥
    # api_key = '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'  # 在这里设置你的OpenRouter API密钥，或使用环境变量OPENROUTER_API_KEY
    api_key = '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'  # 在这里设置你的OpenRouter API密钥，或使用环境变量OPENROUTER_API_KEY
    # api_key = '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'  # 在这里设置你的OpenRouter API密钥，或使用环境变量OPENROUTER_API_KEY

    # 2. 数据集列表
    datasets = [
        "data/NYT_sampled.csv",
        # "data/20NG_Dataset_final.csv",
        # "data/IMDB_Dataset_processed.csv",
        # "data/NeurIPS_Dataset_processed.csv",
        # "data/WikiText_Dataset_clean.csv",
    ]
      #免费models，用于测试
#     models = [
#     # "google/gemini-2.5-pro-exp-03-25", #不存在
#     "google/gemini-2.0-flash-exp:free",  
#     # "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",   #think#不存在
#     "meta-llama/llama-3.3-70b-instruct:free", 
#     "meta-llama/llama-3.2-3b-instruct:free",
#     "moonshotai/kimi-dev-72b:free",    #think
#     "moonshotai/kimi-k2:free",  
#     "tngtech/deepseek-r1t2-chimera:free",
#     "deepseek/deepseek-r1-0528:free",
#     "qwen/qwen3-coder:free",
#     "qwen/qwen3-235b-a22b:free"  #think
# ]
     #最强models
    models = [
    # "openai/gpt-5",  # 可能不可用或需要特殊权限
    # "qwen/qwen3-235b-a22b:free",  # 免费且强大的模型
    # "qwen/qwen3-coder:free",
    # "deepseek/deepseek-chat-v3-0324:free",   #supports both thinking and non-thinking modes via prompt templates.
    # "openai/gpt-5-mini",
    # "openai/gpt-4o-2024-11-20",
    # "google/gemini-2.5-pro-preview",
    # "anthropic/claude-sonnet-4",
    # "anthropic/claude-opus-4.1",
    # "meta-llama/llama-4-maverick",
    "deepseek/deepseek-chat-v3.1",   #supports both thinking and non-thinking modes via prompt templates.
    # "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",   #think
    # "meta-llama/llama-3.3-70b-instruct:free",
    # # "meta-llama/llama-3.2-3b-instruct:free",
]
    
    # 4. 输出目录
    outdir = "llm_analysis/results"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # ================================

    # 批量推理统计
    total_combinations = len(datasets) * len(models)
    current_combination = 0
    successful_runs = 0
    failed_runs = 0
    total_cost = 0.0
    
    print(f"开始批量推理：{len(datasets)} 个数据集 × {len(models)} 个模型 = {total_combinations} 个组合")
    print("=" * 80)
    
    # 遍历每个数据集和每个模型的组合
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).stem  # 获取文件名（不含扩展名）

        for model in models:
            current_combination += 1
            model_name_clean = model.replace("/", "_").replace("-", "_").replace(":", "")
            output_filename = f"topic_analysis_{dataset_name}_{model_name_clean}.json"
            output_path = os.path.join(outdir, output_filename)

            print(f"\n[{current_combination}/{total_combinations}] 处理组合: {dataset_name} + {model}")

            # 检查结果文件是否已存在
            if os.path.exists(output_path):
                print(f"跳过：结果文件已存在 - {output_filename}")
                successful_runs += 1
                continue

            # 检查数据集文件是否存在
            if not Path(dataset_path).exists():
                print(f"错误: 数据集文件不存在 - {dataset_path}")
                failed_runs += 1
                continue

            try:
                # 初始化OpenRouter客户端
                api_client = OpenRouterClient(api_key=api_key, model=model)

                # 初始化主题分析器
                analyzer = TopicAnalyzer(api_client)

                print(f"正在分析: {dataset_path}")
                print(f"使用模型: {model}")

                # 执行分析
                results = analyzer.analyze_text(dataset_path)

                # 保存结果
                analyzer.save_results(results, output_path)

                # 统计成本
                if 'cost_analysis' in results:
                    cost_analysis = results['cost_analysis']
                    run_cost = cost_analysis['actual_cost']
                    total_cost += run_cost

                    print(f"✓ 分析完成！生成 {results['total_topics']} 个主题")
                    print(f"  输入tokens: {cost_analysis['input_tokens']:,}")
                    print(f"  输出tokens: {cost_analysis['output_tokens']:,}")
                    print(f"  本次成本: ${run_cost:.6f}")
                    print(f"  累计成本: ${total_cost:.6f}")
                else:
                    print(f"✓ 分析完成！生成 {results['total_topics']} 个主题")
                    print(f"  使用免费模型，无成本")

                print(f"  结果已保存: {output_filename}")
                successful_runs += 1

            except Exception as e:
                print(f"✗ 分析失败: {str(e)}")
                failed_runs += 1

                # 如果是API相关错误，记录详细信息
                if "401" in str(e) or "402" in str(e) or "429" in str(e):
                    print(f"  API错误，可能需要检查密钥或余额")
                elif "timeout" in str(e).lower():
                    print(f"  请求超时，可能是网络问题")

                continue
    
    # 打印最终统计
    print("\n" + "=" * 80)
    print("批量推理完成！")
    print(f"总组合数: {total_combinations}")
    print(f"成功运行: {successful_runs}")
    print(f"失败运行: {failed_runs}")
    print(f"成功率: {successful_runs/total_combinations*100:.1f}%")
    if total_cost > 0:
        print(f"总成本: ${total_cost:.6f}")
        print(f"平均每次成本: ${total_cost/successful_runs:.6f}")
    else:
        print("总成本: $0.00 (使用免费模型)")
    print(f"结果保存目录: {outdir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
