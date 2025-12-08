import pandas as pd


def diagnose_model(csv_path='../../experiments/EXP_20251208_151742/metrics.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {csv_path}")
        return

    # 1. 找到验证集上 Recall@20 最高的那个 Epoch
    # 假设 csv 中有 phase 列区分 train/val，如果没有则默认全部
    if 'phase' in df.columns:
        val_df = df[df['phase'] == 'val'].copy()
    else:
        val_df = df.copy()

    if val_df.empty:
        print("未找到验证集数据，请检查 metrics.csv")
        return

    best_idx = val_df['Recall@20'].idxmax()
    best_row = val_df.loc[best_idx]

    # 2. 提取关键指标
    print("=" * 40)
    print(f"最佳模型 (Epoch {best_row['epoch']:.0f}) 诊断报告")
    print("=" * 40)
    print(f"核心指标 Recall@20: \t{best_row['Recall@20']:.4f}")
    print(f"NDCG@20:        \t{best_row['NDCG@20']:.4f}")

    # 3. 诊断：是否过拟合热门物品？
    # 读取第一轮和最佳轮的 Popularity
    start_pop = val_df.iloc[0].get('Avg_Popularity@10', 0)
    end_pop = best_row.get('Avg_Popularity@10', 0)

    print("-" * 20)
    print(f"热门度趋势 (Avg_Popularity@10): {start_pop:.4f} -> {end_pop:.4f}")
    if end_pop > start_pop * 1.2:
        print("⚠️ 警告：模型越来越倾向于推荐热门物品 (Popularity Bias 严重)")
        diagnosis = "A"
    elif end_pop > 0.5:  # 假设归一化后的阈值，具体视数据而定
        print("⚠️ 警告：推荐结果整体偏热门")
        diagnosis = "A"
    else:
        print("✅ 热门度控制尚可")
        diagnosis = "B"

    # 4. 诊断：是否推荐过于单一？
    coverage = best_row.get('coverage@10', 0)
    diversity = best_row.get('Cat_Diversity@10', 0)
    print(f"覆盖率 (Coverage@10): \t{coverage:.4f}")
    print(f"多样性 (Diversity@10): \t{diversity:.4f}")

    if coverage < 0.1:  # 举例阈值
        print("⚠️ 警告：模型只覆盖了极少部分物品 (信息茧房风险)")
        if diagnosis == "B": diagnosis = "C"

    print("=" * 40)
    return diagnosis


if __name__ == "__main__":
    diagnosis = diagnose_model()

    print("\n【下一步建议】")
    if diagnosis == "A":
        print("👉 你的模型在偷懒。优先实施【方案2：全局热门采样】。")
        print("   将热门物品强制作为负样本，惩罚模型只推热门的行为。")
    elif diagnosis == "C":
        print("👉 模型陷入了局部最优。优先实施【方案3：模型挖掘困难负样本】。")
        print("   模型可能只学会了区分“极热”和“极冷”，需要更难的样本来逼它细分兴趣。")
    else:
        print("👉 指标看起来比较均衡。可以直接尝试【方案3：模型挖掘困难负样本】来冲击更高分。")