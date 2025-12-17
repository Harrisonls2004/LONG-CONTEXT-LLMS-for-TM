#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有LLM主题建模评估指标

这个脚本会依次运行所有6个独立的指标评估文件
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, Any

# 指标文件列表
METRIC_FILES = [
    {
        "name": "主题下平均分配文本数量",
        "file": "metric_topic_distribution.py",
        "description": "评估每个主题下分配的文本数量分布"
    },
    {
        "name": "高频主题优先生成",
        "file": "metric_high_frequency_priority.py", 
        "description": "判断是否优先生成高频主题"
    },
    {
        "name": "输入文本忽视比（前30%）",
        "file": "metric_input_neglect.py",
        "description": "评估前30%输入文本的忽视情况"
    },
    {
        "name": "指令约束下主题数量上限",
        "file": "metric_max_topics.py",
        "description": "测量LLM可生成的最大主题数量"
    },
    {
        "name": "主题文本相关性/幻觉率",
        "file": "metric_topic_relevance.py",
        "description": "评估主题关键词与文本的相关性"
    },
    {
        "name": "主题重复率/多样性",
        "file": "metric_topic_diversity.py",
        "description": "分析主题的重复率和多样性"
    }
]

def run_single_metric(metric_file: str, metric_name: str) -> Dict[str, Any]:
    """
    运行单个指标评估
    
    Args:
        metric_file (str): 指标文件名
        metric_name (str): 指标名称
        
    Returns:
        Dict: 运行结果
    """
    print(f"\n{'='*60}")
    print(f"🚀 运行指标: {metric_name}")
    print(f"📁 文件: {metric_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行指标文件
        result = subprocess.run(
            [sys.executable, metric_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {metric_name} 运行成功")
            print(f"⏱️ 耗时: {duration:.2f} 秒")
            
            # 尝试读取结果文件
            result_files = {
                "metric_topic_distribution.py": "topic_distribution_results.json",
                "metric_high_frequency_priority.py": "high_frequency_priority_results.json",
                "metric_input_neglect.py": "input_neglect_results.json",
                "metric_max_topics.py": "max_topics_results.json",
                "metric_topic_relevance.py": "topic_relevance_results.json",
                "metric_topic_diversity.py": "topic_diversity_results.json"
            }

            result_file = result_files.get(metric_file)
            metric_result = None

            if result_file:
                # 构建完整路径
                result_dir = "C:/Users/1/Desktop/TopMost/metric_result"
                full_result_path = os.path.join(result_dir, result_file)

                if os.path.exists(full_result_path):
                    try:
                        with open(full_result_path, 'r', encoding='utf-8') as f:
                            metric_result = json.load(f)
                    except Exception as e:
                        print(f"⚠️ 读取结果文件失败: {str(e)}")
            
            return {
                "metric_name": metric_name,
                "file": metric_file,
                "status": "success",
                "duration": duration,
                "stdout": result.stdout,
                "result_data": metric_result
            }
        else:
            print(f"❌ {metric_name} 运行失败")
            print(f"错误输出: {result.stderr}")
            
            return {
                "metric_name": metric_name,
                "file": metric_file,
                "status": "failed",
                "duration": duration,
                "error": result.stderr,
                "stdout": result.stdout
            }
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ {metric_name} 运行异常: {str(e)}")
        
        return {
            "metric_name": metric_name,
            "file": metric_file,
            "status": "error",
            "duration": duration,
            "error": str(e)
        }

def generate_summary_report(results: list) -> Dict[str, Any]:
    """
    生成汇总报告
    
    Args:
        results (list): 所有指标的运行结果
        
    Returns:
        Dict: 汇总报告
    """
    successful_metrics = [r for r in results if r['status'] == 'success']
    failed_metrics = [r for r in results if r['status'] != 'success']
    
    total_duration = sum(r['duration'] for r in results)
    
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_metrics": len(results),
        "successful_metrics": len(successful_metrics),
        "failed_metrics": len(failed_metrics),
        "success_rate": len(successful_metrics) / len(results) if results else 0,
        "total_duration": round(total_duration, 2),
        "average_duration": round(total_duration / len(results), 2) if results else 0,
        "metric_results": results,
        "summary_statistics": {}
    }
    
    # 提取关键统计信息
    for result in successful_metrics:
        if result.get('result_data'):
            metric_name = result['metric_name']
            data = result['result_data']
            
            # 根据不同指标提取关键信息
            if "主题下平均分配文本数量" in metric_name:
                summary['summary_statistics']['topic_distribution'] = {
                    "mean_texts_per_topic": data.get('mean_texts_per_topic'),
                    "distribution_uniformity": data.get('distribution_uniformity')
                }
            elif "高频主题优先生成" in metric_name:
                summary['summary_statistics']['high_frequency_priority'] = {
                    "priority_score": data.get('priority_score'),
                    "stability_score": data.get('stability_score'),
                    "overall_assessment": data.get('evaluation', {}).get('overall_assessment')
                }
            elif "输入文本忽视比" in metric_name:
                summary['summary_statistics']['input_neglect'] = {
                    "front_30_percent_ratio": data.get('front_30_percent_ratio'),
                    "neglect_score": data.get('neglect_score'),
                    "overall_assessment": data.get('position_analysis', {}).get('overall_assessment')
                }
            elif "主题数量上限" in metric_name:
                summary['summary_statistics']['max_topics'] = {
                    "max_topics_generated": data.get('max_topics_generated'),
                    "generation_capacity": data.get('evaluation', {}).get('generation_capacity')
                }
            elif "主题文本相关性" in metric_name:
                summary['summary_statistics']['topic_relevance'] = {
                    "mean_relevance_score": data.get('mean_relevance_score'),
                    "mean_hallucination_ratio": data.get('mean_hallucination_ratio'),
                    "overall_assessment": data.get('evaluation', {}).get('overall_assessment')
                }
            elif "主题重复率" in metric_name:
                summary['summary_statistics']['topic_diversity'] = {
                    "diversity_score": data.get('diversity_score'),
                    "redundancy_rate": data.get('redundancy_rate'),
                    "overall_assessment": data.get('evaluation', {}).get('overall_assessment')
                }
    
    return summary

def main():
    """主函数"""
    print("🚀 LLM主题建模评估指标 - 批量运行")
    print("="*80)
    print(f"📊 将运行 {len(METRIC_FILES)} 个评估指标")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示指标列表
    print(f"\n📋 指标列表:")
    for i, metric in enumerate(METRIC_FILES, 1):
        print(f"   {i}. {metric['name']} - {metric['description']}")
    
    # 询问是否继续
    response = input(f"\n是否继续运行所有指标？(y/n): ").strip().lower()
    if response not in ['y', 'yes', '是']:
        print("❌ 用户取消运行")
        return
    
    # 运行所有指标
    results = []
    
    for i, metric in enumerate(METRIC_FILES, 1):
        print(f"\n📈 进度: {i}/{len(METRIC_FILES)}")
        
        result = run_single_metric(metric['file'], metric['name'])
        results.append(result)
        
        # 短暂暂停避免资源冲突
        if i < len(METRIC_FILES):
            time.sleep(1)
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("📊 生成汇总报告...")
    
    summary_report = generate_summary_report(results)
    
    # 保存汇总报告
    report_file = f"llm_metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        # 确保输出目录存在
        output_dir = "C:/Users/1/Desktop/TopMost/metric_result"
        os.makedirs(output_dir, exist_ok=True)

        # 构建完整路径
        full_report_path = os.path.join(output_dir, report_file)

        with open(full_report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        print(f"💾 汇总报告已保存到: {full_report_path}")
    except Exception as e:
        print(f"❌ 保存汇总报告失败: {str(e)}")
    
    # 显示运行结果
    print(f"\n🎉 批量运行完成！")
    print(f"📊 总指标数: {summary_report['total_metrics']}")
    print(f"✅ 成功运行: {summary_report['successful_metrics']}")
    print(f"❌ 运行失败: {summary_report['failed_metrics']}")
    print(f"📈 成功率: {summary_report['success_rate']:.1%}")
    print(f"⏱️ 总耗时: {summary_report['total_duration']:.2f} 秒")
    
    # 显示失败的指标
    if summary_report['failed_metrics'] > 0:
        print(f"\n❌ 失败的指标:")
        for result in results:
            if result['status'] != 'success':
                print(f"   - {result['metric_name']}: {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 详细结果请查看各个指标生成的结果文件和汇总报告: {full_report_path}")

if __name__ == "__main__":
    main()
