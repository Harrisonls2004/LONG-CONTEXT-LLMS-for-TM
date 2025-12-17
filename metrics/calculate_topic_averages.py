#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算ds.json文件中50个主题的coherence、concise、informative平均值
"""

import json
import statistics

def calculate_topic_averages(json_file_path):
    """计算主题评估的平均值"""
    
    print(f"📊 分析文件: {json_file_path}")
    print("="*60)
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取基本信息
        metadata = data.get('metadata', {})
        topic_evaluations = data.get('topic_evaluations', [])
        
        print(f"📋 文件信息:")
        print(f"   模型: {metadata.get('model', 'Unknown')}")
        print(f"   主题总数: {metadata.get('total_topics', len(topic_evaluations))}")
        print(f"   评估时间: {metadata.get('evaluation_time', 'Unknown')}")
        print(f"   实际主题数: {len(topic_evaluations)}")
        print()
        
        # 收集所有评分
        coherence_scores = []
        conciseness_scores = []
        informativity_scores = []
        overall_scores = []
        
        valid_topics = 0
        invalid_topics = 0
        
        for i, topic in enumerate(topic_evaluations, 1):
            try:
                evaluation = topic.get('evaluation', {}).get('evaluation', {})
                
                # 提取各项评分
                coherence = evaluation.get('coherence', {}).get('score')
                conciseness = evaluation.get('conciseness', {}).get('score')
                informativity = evaluation.get('informativity', {}).get('score')
                overall = topic.get('evaluation', {}).get('overall_score')
                
                # 验证评分有效性
                if all(score is not None for score in [coherence, conciseness, informativity, overall]):
                    coherence_scores.append(coherence)
                    conciseness_scores.append(conciseness)
                    informativity_scores.append(informativity)
                    overall_scores.append(overall)
                    valid_topics += 1
                else:
                    print(f"⚠️  主题 {i} 评分数据不完整，跳过")
                    invalid_topics += 1
                    
            except Exception as e:
                print(f"❌ 主题 {i} 数据解析错误: {e}")
                invalid_topics += 1
        
        print(f"📈 数据统计:")
        print(f"   有效主题: {valid_topics}")
        print(f"   无效主题: {invalid_topics}")
        print()
        
        if not coherence_scores:
            print("❌ 没有找到有效的评分数据")
            return None
        
        # 计算平均值和其他统计信息
        results = {
            'coherence': {
                'mean': statistics.mean(coherence_scores),
                'median': statistics.median(coherence_scores),
                'stdev': statistics.stdev(coherence_scores) if len(coherence_scores) > 1 else 0,
                'min': min(coherence_scores),
                'max': max(coherence_scores),
                'count': len(coherence_scores)
            },
            'conciseness': {
                'mean': statistics.mean(conciseness_scores),
                'median': statistics.median(conciseness_scores),
                'stdev': statistics.stdev(conciseness_scores) if len(conciseness_scores) > 1 else 0,
                'min': min(conciseness_scores),
                'max': max(conciseness_scores),
                'count': len(conciseness_scores)
            },
            'informativity': {
                'mean': statistics.mean(informativity_scores),
                'median': statistics.median(informativity_scores),
                'stdev': statistics.stdev(informativity_scores) if len(informativity_scores) > 1 else 0,
                'min': min(informativity_scores),
                'max': max(informativity_scores),
                'count': len(informativity_scores)
            },
            'overall': {
                'mean': statistics.mean(overall_scores),
                'median': statistics.median(overall_scores),
                'stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'min': min(overall_scores),
                'max': max(overall_scores),
                'count': len(overall_scores)
            }
        }
        
        # 显示结果
        print("🎯 评估结果统计:")
        print("="*60)
        
        for metric, stats in results.items():
            print(f"\n📊 {metric.upper()}:")
            print(f"   平均值: {stats['mean']:.3f}")
            print(f"   中位数: {stats['median']:.3f}")
            print(f"   标准差: {stats['stdev']:.3f}")
            print(f"   最小值: {stats['min']}")
            print(f"   最大值: {stats['max']}")
            print(f"   样本数: {stats['count']}")
        
        # 简洁总结
        print(f"\n🎯 简洁总结:")
        print("="*60)
        print(f"Coherence 平均值:    {results['coherence']['mean']:.3f}")
        print(f"Conciseness 平均值:  {results['conciseness']['mean']:.3f}")
        print(f"Informativity 平均值: {results['informativity']['mean']:.3f}")
        print(f"Overall 平均值:      {results['overall']['mean']:.3f}")
        
        # 评估质量等级
        print(f"\n📈 质量评估:")
        print("="*60)
        
        def get_quality_level(score):
            if score >= 4.5:
                return "优秀 (Excellent)"
            elif score >= 4.0:
                return "良好 (Good)"
            elif score >= 3.5:
                return "中等 (Average)"
            elif score >= 3.0:
                return "一般 (Below Average)"
            else:
                return "较差 (Poor)"
        
        print(f"Coherence:    {get_quality_level(results['coherence']['mean'])}")
        print(f"Conciseness:  {get_quality_level(results['conciseness']['mean'])}")
        print(f"Informativity: {get_quality_level(results['informativity']['mean'])}")
        print(f"Overall:      {get_quality_level(results['overall']['mean'])}")
        
        return results
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        return None

def main():
    """主函数"""
    json_file = r"traditional_models/results/ds.json"
    
    print("🔍 主题评估平均值计算器")
    print("="*60)
    
    results = calculate_topic_averages(json_file)
    
    if results:
        print(f"\n✅ 计算完成！")
        
        # 保存结果到文件
        output_file = "topic_averages_summary.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"📁 详细结果已保存到: {output_file}")
        except Exception as e:
            print(f"⚠️  保存结果文件失败: {e}")
    else:
        print(f"\n❌ 计算失败")

if __name__ == "__main__":
    main()
