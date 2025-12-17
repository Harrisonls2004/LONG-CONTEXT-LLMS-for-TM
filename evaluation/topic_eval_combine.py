import json
import os
from pathlib import Path
from typing import Dict, Any, List, Union

from topic_evaluation_Tra import TraTopicEvaluator
from topic_evaluation_LLM import topic_evaluation_LLM
import pandas as pd

class TopicEvaluationRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.texts = self._load_texts()
        self.topics = self._load_topics()
        
    def _load_texts(self) -> list:
        """Load reference texts from file"""
        # Try different encodings to handle various file formats
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'gbk']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(self.config['texts_path'], encoding=encoding)
                print(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode CSV file with any of the tried encodings: {encodings}")
            
        if 'text' not in df.columns:
            raise ValueError("CSV file must contain a 'text' column")
        return df['text'].tolist()
            
    def _load_topics(self) -> list:
        """Load topics from JSON file or convert from keyword list"""
        # Check if topics_data is provided directly as a list
        if 'topics_data' in self.config and self.config['topics_data'] is not None:
            topics_data = self.config['topics_data']
            # If it's a list of lists (keyword lists), convert to topic format
            if isinstance(topics_data, list) and all(isinstance(topic, list) for topic in topics_data):
                return self._convert_keywords_to_topics(topics_data)
            # If it's already in the correct format, return as is
            elif isinstance(topics_data, list) and all(isinstance(topic, dict) for topic in topics_data):
                return topics_data
            else:
                raise ValueError("topics_data must be a list of keyword lists or a list of topic dictionaries")
        
        # Otherwise, load from JSON file
        elif 'topics_path' in self.config:
            with open(self.config['topics_path'], 'r', encoding='utf-8') as f:
                return json.load(f)["topics"]
        
        else:
            raise ValueError("Either 'topics_data' or 'topics_path' must be provided in config")
    
    def _convert_keywords_to_topics(self, keyword_lists: List[List[str]]) -> List[Dict[str, Any]]:
        """Convert keyword lists to topic dictionary format"""
        topics = []
        for i, keywords in enumerate(keyword_lists):
            topic = {
                "id": i,
                "keywords": keywords,
                "name": f"Topic_{i}",
                "description": f"Topic {i} with keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}"
            }
            topics.append(topic)
        return topics
    
    def run_llm_evaluation(self) -> Dict:
        """Run LLM-based topic evaluation"""
        return topic_evaluation_LLM(
            topics_data=self.topics,
            api_key=self.config['api_key'],
            model=self.config['model'],
            output_file=self.config['output_file']
        )
        
    def run_metric_evaluation(self) -> Dict:
        """Run metric-based topic evaluation"""
        evaluator = TraTopicEvaluator(reference_corpus=self.texts)
        return evaluator.evaluate_all(self.topics, topk=5)
        
    def print_results(self, results: Dict) -> None:
        """Print evaluation results"""
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")

def main():
    # Example 1: Using JSON file (original method)
    config_json = {
        'texts_path': "data/NYT_Dataset.csv",
        'topics_path': "llm_analysis/results/1_topic_analysis_NYT_Dataset_deepseek_deepseek_chat_v3_0324free.json",
        'api_key': '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',
        'model': "moonshotai/kimi-k2:free",
        'output_file': None,
    }
    config_json['output_file'] = config_json['topics_path'].replace(".json", "_llm_evaluation_results.json")

    # # Example 2: Using keyword lists (new method)
    # sample_keyword_lists = [
    #     ["政治", "政府", "选举", "政策", "国会"],
    #     ["经济", "市场", "股票", "投资", "金融"],
    #     ["科技", "人工智能", "计算机", "互联网", "创新"],
    #     ["体育", "足球", "篮球", "比赛", "运动员"],
    #     ["娱乐", "电影", "音乐", "明星", "演出"]
    # ]
    
    # config_keywords = {
    #     'texts_path': "data/NYT_Dataset.csv",
    #     'topics_data': sample_keyword_lists,  # 直接提供关键词列表
    #     'api_key': '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',
    #     'model': "moonshotai/kimi-k2:free",
    #     'output_file': "keyword_topics_evaluation_results.json",
    # }

    # # Choose which config to use
    # config = config_keywords  # 使用关键词列表方式
    # # config = config_json    # 或者使用JSON文件方式

    # runner = TopicEvaluationRunner(config)

    # # Run metric evaluation and print results
    # results = runner.run_metric_evaluation()
    # runner.print_results(results)
    
    # # Run LLM evaluation
    # _ = runner.run_llm_evaluation()


if __name__ == "__main__":
    main()
