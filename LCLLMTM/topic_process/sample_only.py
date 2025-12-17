#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

def sample_nyt_data():
    """
    从NYT_Dataset.csv中随机采样14,000行数据，保存为NYT_sampled.csv
    """
    print("🚀 开始NYT数据集采样...")
    
    # 文件路径
    input_file = "../data4LCLLM/NYT_Dataset.csv"
    output_file = "../data4LCLLM/NYT_sampled.csv"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        # 尝试绝对路径
        input_file = "C:/Users/1/Desktop/TopMost/LCLLMTM/data4LCLLM/NYT_Dataset.csv"
        output_file = "C:/Users/1/Desktop/TopMost/LCLLMTM/data4LCLLM/NYT_sampled.csv"
        
        if not os.path.exists(input_file):
            print(f"❌ 绝对路径也不存在: {input_file}")
            return False
    
    try:
        print(f"📁 读取文件: {input_file}")
        
        # 读取数据
        df = pd.read_csv(input_file)
        print(f"📊 原始数据行数: {len(df):,}")
        print(f"📋 数据列: {list(df.columns)}")
        
        # 检查数据量
        sample_size = 14000
        if len(df) < sample_size:
            print(f"⚠️ 数据量不足，调整采样数量为: {len(df):,}")
            sample_size = len(df)
        
        # 随机采样
        print(f"🎲 随机采样 {sample_size:,} 行...")
        np.random.seed(42)  # 设置随机种子确保结果可重现
        sampled_df = df.sample(n=sample_size, random_state=42)
        sampled_df = sampled_df.reset_index(drop=True)
        
        print(f"✅ 采样完成，共 {len(sampled_df):,} 行")
        
        # 保存文件
        print(f"💾 保存到: {output_file}")
        sampled_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 验证保存
        if os.path.exists(output_file):
            # 重新读取验证
            verify_df = pd.read_csv(output_file)
            print(f"\n🎉 采样成功完成!")
            print(f"   📊 保存行数: {len(verify_df):,}")
            print(f"   📈 采样比例: {len(verify_df)/len(df)*100:.1f}%")
            print(f"   📁 输出文件: {output_file}")
            
            # 显示前几行作为验证
            if 'title' in verify_df.columns:
                print(f"\n📰 采样数据预览 (前3行标题):")
                for i in range(min(3, len(verify_df))):
                    title = verify_df.iloc[i]['title']
                    print(f"   {i+1}. {title}")
            
            # 显示文件大小
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   📦 文件大小: {file_size:.1f} MB")
            
            return True
        else:
            print("❌ 文件保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 NYT数据集随机采样工具")
    print("=" * 60)
    
    success = sample_nyt_data()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 任务完成! NYT_sampled.csv 已生成")
    else:
        print("❌ 任务失败!")
        sys.exit(1)
    print("=" * 60)
