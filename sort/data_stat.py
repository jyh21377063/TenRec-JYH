import pandas as pd
import numpy as np
import os


def calculate_target_stats(file_path):
    print(f"Loading data from {file_path}...")

    # 我们只需要读取目标列来做统计，节省内存
    target_cols = ['click', 'like', 'follow', 'share']

    try:
        df = pd.read_csv(file_path, usecols=lambda c: c in target_cols)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    total_samples = len(df)
    print(f"Total samples: {total_samples}\n")
    print("-" * 40)
    print(f"{'Target':<10} | {'Positive':<10} | {'Ratio (%)':<10}")
    print("-" * 40)

    stats = {}
    for col in target_cols:
        if col in df.columns:
            pos_count = df[col].sum()
            ratio = pos_count / total_samples
            stats[col] = ratio
            print(f"{col:<10} | {int(pos_count):<10} | {ratio * 100:.4f}%")
        else:
            print(f"{col:<10} | Not Found in dataset")

    print("-" * 40)

    # --- 计算推荐权重 (平滑倒数策略) ---
    if stats:
        print("\n[ Recommended Manual Weights ]")
        # 找到最高频的任务作为基准 (通常是 click)
        max_ratio = max(stats.values())

        weights = {}
        for col, ratio in stats.items():
            if ratio > 0:
                # 使用平方根平滑倒数权重
                w = np.sqrt(max_ratio / ratio)
                weights[col] = round(w, 4)
            else:
                weights[col] = 0.0

        print(f"Formula: sqrt(max_ratio / target_ratio)")
        for col, w in weights.items():
            print(f"{col:<10} weight: {w}")


if __name__ == '__main__':
    # 替换为你原始数据的实际路径
    csv_path = 'ctr_data_1M.csv'

    if os.path.exists(csv_path):
        calculate_target_stats(csv_path)
    else:
        print(f"File not found: {csv_path}. Please check the path.")


"""    
----------------------------------------
Target     | Positive   | Ratio (%) 
----------------------------------------
click      | 28880860   | 23.9989%
like       | 2275417    | 1.8908%
follow     | 179788     | 0.1494%
share      | 250089     | 0.2078%
----------------------------------------

[ Recommended Manual Weights ]
Formula: sqrt(max_ratio / target_ratio)
click      weight: 1.0
like       weight: 3.5627
follow     weight: 12.6743
share      weight: 10.7463

"""