"""
主题模型评测器
提供简单易用的函数接口来评测主题模型的准确性
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
    
    def _load_data_from_csv(self, filepath):
        """从CSV文件加载文本和关键词数据"""
        try:
            # Use pandas to read CSV file with automatic encoding detection
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # 提取所需的列：text, topic ID, topic word list
            data = []
            for index, row in df.iterrows():
                text = str(row['text'])
                topic_id = str(row['topic ID'])
                keywords_str = str(row['topic word list'])
                keywords = [kw.strip() for kw in keywords_str.split() if kw.strip()]
                
                data.append({
                    'index': index,
                    'text': text,
                    'topic_id': topic_id,
                    'keywords': keywords
                })
            
            return data
        except UnicodeDecodeError:
            try:
                # 如果UTF-8失败，尝试latin-1编码
                df = pd.read_csv(filepath, encoding='latin-1')
                
                data = []
                for index, row in df.iterrows():
                    text = str(row['text'])
                    topic_id = str(row['topic ID'])
                    keywords_str = str(row['topic word list'])
                    keywords = [kw.strip() for kw in keywords_str.split() if kw.strip()]
                    
                    data.append({
                        'index': index,
                        'text': text,
                        'topic_id': topic_id,
                        'keywords': keywords
                    })
                
                return data
            except Exception as e:
                print(f"[ERROR] Failed to load data with latin-1: {e}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return None
    
    def evaluate_topics(self, data_file, output_file=None, progress_callback=None):
        """
        评测主题模型
        
        Args:
            data_file (str): 包含文本和关键词的CSV文件路径
            output_file (str): 输出CSV文件路径（可选）
            progress_callback (callable): 进度回调函数（可选）
            
        Returns:
            dict: 评测结果统计
        """
        # 设置默认输出文件名
        if output_file is None:
            output_file = f"evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 加载数据
        data = self._load_data_from_csv(data_file)
        if data is None:
            raise Exception("Failed to load data")
        
        # 计算总评测数量
        total_evaluations = len(data)
        current_evaluation = 0
        
        # 统计变量
        results = []
        stats = {'total': 0, 'yes': 0, 'no': 0, 'error': 0}
        
        # 创建CSV文件并写入标题行
        f = open(output_file, 'w', newline='', encoding='utf-8')
        print(f"[INFO] Output file: {output_file}")
        writer = csv.writer(f)
        writer.writerow(['source_index', 'text', 'topic_id', 'topic_keywords', 'llm_decision', 'explanation'])
        
        try:
            # 开始评测
            for item in data:
                current_evaluation += 1
                
                index = item['index']
                text = item['text']
                topic_id = item['topic_id']
                keywords = item['keywords']
                
                # 调用进度回调
                if progress_callback:
                    progress = current_evaluation / total_evaluations * 100
                    progress_callback(current_evaluation, total_evaluations, progress, topic_id, index)
                
                # 获取LLM判断
                decision, explanation = self._get_llm_decision(text, keywords)
                
                # 保存结果到内存
                results.append({
                    'source_index': index,
                    'text': text,
                    'topic_id': topic_id,
                    'keywords': keywords,
                    'decision': decision,
                    'explanation': explanation
                })
                
                # 立即写入CSV文件并刷新
                writer.writerow([index, text, topic_id, ', '.join(keywords), decision, explanation])
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
            topic_id = result['topic_id']
            decision = result['decision']
            
            if topic_id not in topic_stats:
                topic_stats[topic_id] = {'total': 0, 'yes': 0, 'no': 0, 'error': 0}
            topic_stats[topic_id]['total'] += 1
            topic_stats[topic_id][decision] += 1
        
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
            for topic_id in sorted(topic_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                topic_stat = topic_stats[topic_id]
                total = topic_stat['total']
                yes_pct = topic_stat['yes'] / total * 100 if total > 0 else 0
                no_pct = topic_stat['no'] / total * 100 if total > 0 else 0
                error_pct = topic_stat['error'] / total * 100 if total > 0 else 0
                f.write(f"主题 #{topic_id}: 总数={total}, YES={topic_stat['yes']}({yes_pct:.1f}%), "
                       f"NO={topic_stat['no']}({no_pct:.1f}%), ERROR={topic_stat['error']}({error_pct:.1f}%)\n")
                f.flush()  # 确保数据立即写入文件


def evaluate_topics(data_file, api_key, output_file=None, progress_callback=None):
    """
    便捷的主题评测函数
    
    Args:
        data_file (str): 包含文本和关键词的CSV文件路径
        api_key (str): OpenRouter API密钥
        output_file (str): 输出文件路径（可选）
        progress_callback (callable): 进度回调函数（可选）
        
    Returns:
        dict: 评测结果统计
        
    Example:
        >>> def show_progress(current, total, percent, topic_id, doc_index):
        ...     print(f"Progress: {current}/{total} ({percent:.1f}%) - Topic #{topic_id}")
        ...
        >>> result = evaluate_topics(
        ...     'data.csv', 
        ...     'your_api_key',
        ...     progress_callback=show_progress
        ... )
        >>> print(f"Evaluation completed: {result['yes_rate']:.1f}% match rate")
    """
    evaluator = TopicEvaluator(api_key)
    return evaluator.evaluate_topics(data_file, output_file, progress_callback)


if __name__ == "__main__":
    # 使用示例
    def progress_callback(current, total, percent, topic_id, doc_index):
        print(f"[PROGRESS] {current}/{total} ({percent:.1f}%) - Topic #{topic_id}, Doc #{doc_index}")
    
    # 配置信息
    API_KEY = "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"
    # API_KEY = "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"
    DATA_FILE = 'traditional_models/results/DecTM_testdata_100.csv'
    
    try:
        result = evaluate_topics(
            data_file=DATA_FILE,
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
