"""
LLM主题分配准确率评测器
测试LLM分析得出的主题分配是否准确

功能：
1. 读取LLM主题分析结果文件（包含source_titles_with_ids）
2. 根据文章ID从源数据中获取对应的文本内容
3. 使用LLM判断这些文本是否真的属于对应的主题
4. 生成详细的评测报告和统计结果

使用方法：
    from topic_evaluator import evaluate_llm_topic_assignment

    result = evaluate_llm_topic_assignment(
        llm_result_file='path/to/llm_topic_analysis.json',
        source_csv_file='path/to/NYT_Dataset.csv',
        api_key='your_openrouter_api_key'
    )

    print(f"主题分配准确率: {result['yes_rate']:.1f}%")
"""

import json
import csv
import os
import datetime
import pandas as pd
from openai import OpenAI


class TopicEvaluator:
    """主题模型评测器类"""
    
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1"):
        """
        初始化评测器
        
        Args:
            api_key (str): OpenRouter API密钥
            base_url (str): API基础URL
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/topic-evaluator",
                "X-Title": "Topic Evaluator",
            },
        )
        
    def _get_optimized_prompt(self, text, keywords):
        """生成优化的主题分类提示词"""
        return f"""You are an expert topic classifier. Your task is to determine whether a given text belongs to a specific topic defined by a set of keywords.

TEXT: {text}

TOPIC KEYWORDS: {', '.join(keywords)}

Analyze whether this text is related to the topic represented by these keywords. Provide your response in the following JSON format:
{{
  "answer": "yes" or "no",
  "explanation": "Brief explanation of your reasoning"
}}
IMPORTANT: Respond ONLY with valid JSON in the exact format above."""

    def _get_llm_decision(self, text, keywords):
        """
        调用LLM API进行主题判断
        
        Args:
            text (str): 待评估的文本
            keywords (list): 主题关键词列表
            
        Returns:
            tuple: (decision, explanation) - ('yes'/'no'/'error', explanation_text)
        """
        try:
            prompt = self._get_optimized_prompt(text, keywords)
            response = self.client.chat.completions.create(
                model="qwen/qwen3-14b:free",
                # model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON响应
            try:
                import json
                response_json = json.loads(response_text)
                answer = response_json.get('answer', '').lower()
                explanation = response_json.get('explanation', '')
                
                if answer in ['yes', 'no']:
                    return answer, explanation
                else:
                    return 'no', explanation  # 默认为no
                    
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试从文本中提取答案
                response_lower = response_text.lower()
                if 'yes' in response_lower:
                    return 'yes', response_text
                elif 'no' in response_lower:
                    return 'no', response_text
                else:
                    return 'no', response_text
                
        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}")
            return 'error', str(e)
    
    def _load_source_texts(self, filepath):
        """从CSV文件加载源文本，返回ID到文本的映射"""
        try:
            # Use pandas to read CSV file with automatic encoding detection
            df = pd.read_csv(filepath, encoding='utf-8')
            print("Successfully loaded CSV with utf-8 encoding")

            # 创建ID到文本的映射
            # 假设CSV有'id'和'text'列，如果没有id列则使用索引
            if 'id' in df.columns:
                id_to_text = dict(zip(df['id'].astype(str), df['text'].astype(str)))
            else:
                # 如果没有id列，使用行索引作为ID
                id_to_text = dict(zip(df.index.astype(str), df['text'].astype(str)))

            return id_to_text
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试latin-1编码
                df = pd.read_csv(filepath, encoding='latin-1')
                print("Successfully loaded CSV with latin-1 encoding")

                if 'id' in df.columns:
                    id_to_text = dict(zip(df['id'].astype(str), df['text'].astype(str)))
                else:
                    id_to_text = dict(zip(df.index.astype(str), df['text'].astype(str)))

                return id_to_text
            except Exception as e:
                print(f"[ERROR] Failed to load source texts with latin-1: {e}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to load source texts: {e}")
            return None
    
    def evaluate_topics(self, topics_file, source_file, output_file=None, progress_callback=None):
        """
        评测主题模型 - 测试LLM分配的文本是否真的属于对应主题

        Args:
            topics_file (str): LLM主题分析结果JSON文件路径
            source_file (str): 源数据CSV文件路径
            output_file (str): 输出CSV文件路径（可选）
            progress_callback (callable): 进度回调函数（可选）

        Returns:
            dict: 评测结果统计
        """
        # 设置默认输出文件名
        if output_file is None:
            output_file = f"topic_accuracy_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 加载主题数据
        try:
            with open(topics_file, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load topics file: {e}")

        # 加载源文本ID映射
        id_to_text = self._load_source_texts(source_file)
        if id_to_text is None:
            raise Exception("Failed to load source texts")

        # 计算总评测数量 - 基于source_titles_with_ids
        total_evaluations = sum(len(topic.get('source_titles_with_ids', [])) for topic in topics_data.get('topics', []))
        current_evaluation = 0
        
        # 统计变量
        results = []
        stats = {'total': 0, 'yes': 0, 'no': 0, 'error': 0}
        
        # 创建CSV文件并写入标题行
        f = open(output_file, 'w', newline='', encoding='utf-8')
        print(f"[INFO] Output file: {output_file}")
        writer = csv.writer(f)
        writer.writerow(['article_id', 'article_title', 'text', 'topic_num', 'topic_summary', 'topic_keywords', 'llm_decision', 'explanation'])

        try:
            # 开始评测
            for topic in topics_data.get('topics', []):
                topic_num = topic.get('topic_num')
                topic_summary = topic.get('summary', '')
                keywords = topic.get('keywords', [])

                # 使用source_titles_with_ids而不是source_indices
                for source_item in topic.get('source_titles_with_ids', []):
                    current_evaluation += 1

                    # 获取文章ID和标题
                    article_id = str(source_item.get('id', ''))
                    article_title = source_item.get('title', '')

                    # 根据ID获取文本内容
                    if article_id not in id_to_text:
                        print(f"[WARNING] Article ID {article_id} not found in source data")
                        continue

                    text = id_to_text[article_id]

                    # 调用进度回调
                    if progress_callback:
                        progress = current_evaluation / total_evaluations * 100
                        progress_callback(current_evaluation, total_evaluations, progress, topic_num, article_id)

                    # 获取LLM判断
                    decision, explanation = self._get_llm_decision(text, keywords)

                    # 保存结果到内存
                    results.append({
                        'article_id': article_id,
                        'article_title': article_title,
                        'text': text,
                        'topic_num': topic_num,
                        'topic_summary': topic_summary,
                        'keywords': keywords,
                        'decision': decision,
                        'explanation': explanation
                    })

                    # 立即写入CSV文件并刷新
                    writer.writerow([article_id, article_title, text, topic_num, topic_summary, ', '.join(keywords), decision, explanation])
                    f.flush()  # 确保数据立即写入文件

                    # 更新统计
                    stats['total'] += 1
                    stats[decision] += 1
        finally:
            # 确保文件被关闭
            f.close()
            print(f"[INFO] CSV file closed: {output_file}")
        
        # 生成分析报告
        self._generate_report(results, stats, output_file)
        
        return {
            'output_file': output_file,
            'total_evaluations': stats['total'],
            'yes_count': stats['yes'],
            'no_count': stats['no'],
            'error_count': stats['error'],
            'yes_rate': stats['yes'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'no_rate': stats['no'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'error_rate': stats['error'] / stats['total'] * 100 if stats['total'] > 0 else 0
        }
    
    def _generate_report(self, results, stats, output_file):
        """生成详细分析报告"""
        report_file = output_file.replace('.csv', '_report.txt')
        print({report_file})
        # 按主题统计
        topic_stats = {}
        for result in results:
            topic_num = result['topic_num']
            decision = result['decision']
            
            if topic_num not in topic_stats:
                topic_stats[topic_num] = {'total': 0, 'yes': 0, 'no': 0, 'error': 0}
            topic_stats[topic_num]['total'] += 1
            topic_stats[topic_num][decision] += 1
        
        # 使用with语句确保文件正确关闭
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"主题模型评测报告\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"使用优化提示词进行自动评测\n\n")
            
            f.write(f"总体统计:\n")
            f.write(f"- 总评测数量: {stats['total']}\n")
            f.write(f"- 匹配主题: {stats['yes']} ({stats['yes']/stats['total']*100:.1f}%)\n")
            f.write(f"- 不匹配主题: {stats['no']} ({stats['no']/stats['total']*100:.1f}%)\n")
            f.write(f"- API错误: {stats['error']} ({stats['error']/stats['total']*100:.1f}%)\n\n")
            
            f.write(f"各主题详细统计:\n")
            for topic_num in sorted(topic_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                topic_stat = topic_stats[topic_num]
                total = topic_stat['total']
                yes_pct = topic_stat['yes'] / total * 100 if total > 0 else 0
                no_pct = topic_stat['no'] / total * 100 if total > 0 else 0
                error_pct = topic_stat['error'] / total * 100 if total > 0 else 0
                f.write(f"主题 #{topic_num}: 总数={total}, YES={topic_stat['yes']}({yes_pct:.1f}%), "
                       f"NO={topic_stat['no']}({no_pct:.1f}%), ERROR={topic_stat['error']}({error_pct:.1f}%)\n")
                f.flush()  # 确保数据立即写入文件


def evaluate_topics(topics_file, source_file, api_key, output_file=None, progress_callback=None):
    """
    便捷的主题评测函数 - 测试LLM主题分配的准确性

    Args:
        topics_file (str): LLM主题分析结果JSON文件路径
        source_file (str): 源数据CSV文件路径
        api_key (str): OpenRouter API密钥
        output_file (str): 输出文件路径（可选）
        progress_callback (callable): 进度回调函数（可选）

    Returns:
        dict: 评测结果统计

    Example:
        >>> def show_progress(current, total, percent, topic_num, article_id):
        ...     print(f"Progress: {current}/{total} ({percent:.1f}%) - Topic #{topic_num}, Article #{article_id}")
        ...
        >>> result = evaluate_topics(
        ...     'llm_topic_analysis_results.json',
        ...     'NYT_Dataset.csv',
        ...     'your_api_key',
        ...     progress_callback=show_progress
        ... )
        >>> print(f"Topic assignment accuracy: {result['yes_rate']:.1f}%")
    """
    evaluator = TopicEvaluator(api_key)
    return evaluator.evaluate_topics(topics_file, source_file, output_file, progress_callback)


def evaluate_llm_topic_assignment(llm_result_file, source_csv_file, api_key):
    """
    快速评测LLM主题分配准确率的便捷函数

    Args:
        llm_result_file (str): LLM主题分析结果文件路径
        source_csv_file (str): 源数据CSV文件路径
        api_key (str): OpenRouter API密钥

    Returns:
        dict: 包含准确率等统计信息的结果
    """
    def progress_callback(current, total, percent, topic_num, article_id):
        print(f"[PROGRESS] {current}/{total} ({percent:.1f}%) - Topic #{topic_num}, Article #{article_id}")

    return evaluate_topics(
        topics_file=llm_result_file,
        source_file=source_csv_file,
        api_key=api_key,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # 使用示例
    def progress_callback(current, total, percent, topic_num, article_id):
        print(f"[PROGRESS] {current}/{total} ({percent:.1f}%) - Topic #{topic_num}, Article ID #{article_id}")

    # 配置信息
    API_KEY = "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"

    # 修改为您要测试的LLM主题分析结果文件
    TOPICS_FILE = 'llm_analysis/results/topic_analysis_NYT_Dataset_deepseek_deepseek_chat_v3_0324free.json'
    SOURCE_FILE = 'data/NYT_Dataset.csv'
    OUTPUT_FILE = 'evaluation/results/topic_accuracy_evaluation_sampled_llama_llama_4_maverick.csv'
    
    try:
        result = evaluate_topics(
            topics_file=TOPICS_FILE,
            source_file=SOURCE_FILE,
            api_key=API_KEY,
            progress_callback=progress_callback
        )
        
        print(f"\n=== 评测完成 ===")
        print(f"输出文件: {result['output_file']}")
        print(f"总评测数量: {result['total_evaluations']}")
        print(f"匹配率: {result['yes_rate']:.1f}%")
        print(f"不匹配率: {result['no_rate']:.1f}%")
        print(f"错误率: {result['error_rate']:.1f}%")
        
    except Exception as e:
        print(f"[ERROR] 评测失败: {e}")
