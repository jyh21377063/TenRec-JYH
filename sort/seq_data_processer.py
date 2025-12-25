import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

# 输入文件路径 (请确保路径正确)
INPUT_FILE = 'ctr_data_1M.csv'
# 输出目录
OUTPUT_DIR = 'seq_data_din'

# 序列长度：增加到 50
MAX_SEQ_LEN = 50

# 指定读取的列，防止读取不必要的列浪费内存
USE_COLS = ['user_id', 'item_id', 'click', 'like', 'follow', 'share', 'gender', 'age', 'video_category']


def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. 读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE, usecols=lambda c: c in USE_COLS)
    except ValueError:
        print("列名匹配失败，尝试读取所有列再筛选...")
        df = pd.read_csv(INPUT_FILE)
        # 只要核心列 (这里假设 label 列名是固定的)
        target_cols = ['click', 'like', 'follow', 'share']
        # 找出存在于 csv 中的列
        available_cols = [c for c in USE_COLS if c in df.columns]
        df = df[available_cols]

    print(f"原始数据行数: {len(df)}")

    # 填充缺失值 (以防万一)
    # 对于类别特征，NaN 视为一个新类别
    df.fillna(0, inplace=True)

    print("2. 特征编码 & 维度计算...")

    feature_dims = {}

    # --- A. User ID ---
    lbe_user = LabelEncoder()
    df['user_id'] = lbe_user.fit_transform(df['user_id'])
    feature_dims['user_id'] = df['user_id'].max() + 1

    # --- B. Item ID (特殊处理：0留给padding) ---
    lbe_item = LabelEncoder()
    df['item_id'] = lbe_item.fit_transform(df['item_id']) + 1
    feature_dims['item_id'] = df['item_id'].max() + 1

    # --- C. 其他类别特征 (Age, Gender, Category...) ---
    # 排除 ID 和 Label
    exclude_cols = ['user_id', 'item_id', 'click', 'like', 'follow', 'share']
    other_sparse_cols = [c for c in df.columns if c not in exclude_cols]

    for col in other_sparse_cols:
        print(f"   正在编码: {col} ...")
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col])
        # 记录维度
        feature_dims[col] = df[col].max() + 1
        print(f"   -> {col} dim: {feature_dims[col]}")

    user_num = feature_dims['user_id']
    item_num = feature_dims['item_id']
    print(f"User Num: {user_num}, Item Num: {item_num}")

    # ----------------------------------------------------
    # 3. 生成序列 (保持不变)
    # ----------------------------------------------------
    print(f"3. 生成动态历史序列 (Max Len = {MAX_SEQ_LEN})...")

    # 转换为 numpy 加速
    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    clicks = df['click'].values if 'click' in df.columns else np.zeros(len(df))

    # 结果容器
    seqs = np.zeros((len(df), MAX_SEQ_LEN), dtype=np.int32)
    seq_lens = np.zeros(len(df), dtype=np.int32)

    user_history_map = {}

    for i in tqdm(range(len(df)), desc="Generating Sequences"):
        u = user_ids[i]
        item = item_ids[i]
        is_click = clicks[i]

        hist = user_history_map.get(u, [])
        hist_len = len(hist)
        seq_lens[i] = min(hist_len, MAX_SEQ_LEN)

        if hist_len > 0:
            if hist_len <= MAX_SEQ_LEN:
                # Pre-padding: [0, 0, 1, 2] -> 适合 RNN/Transformer/DIN
                seqs[i, -hist_len:] = hist
            else:
                seqs[i] = hist[-MAX_SEQ_LEN:]

        if is_click == 1:
            if u not in user_history_map:
                user_history_map[u] = []
            user_history_map[u].append(item)

    # ----------------------------------------------------
    # 4. 保存数据
    # ----------------------------------------------------
    print("4. 保存处理后的数据...")

    # 提取 Sparse Features (包括 user_id, item_id, age, gender...)
    # 这里的顺序很重要，之后 dataset.py 会按这个顺序读取
    target_names = ['click', 'like', 'follow', 'share']
    sparse_cols = [c for c in df.columns if c not in target_names]

    print(f"最终 Sparse Feature 列表: {sparse_cols}")

    X_sparse = df[sparse_cols].values

    # 提取 Labels
    for t in target_names:
        if t not in df.columns:
            df[t] = 0
    Y = df[target_names].values

    # 划分 Train/Val/Test
    total = len(df)
    train_idx = int(total * 0.8)
    val_idx = int(total * 0.9)

    # 保存 Train
    np.save(os.path.join(OUTPUT_DIR, 'train_sparse.npy'), X_sparse[:train_idx])
    np.save(os.path.join(OUTPUT_DIR, 'train_seq.npy'), seqs[:train_idx])
    np.save(os.path.join(OUTPUT_DIR, 'train_y.npy'), Y[:train_idx])

    # 保存 Val
    np.save(os.path.join(OUTPUT_DIR, 'val_sparse.npy'), X_sparse[train_idx:val_idx])
    np.save(os.path.join(OUTPUT_DIR, 'val_seq.npy'), seqs[train_idx:val_idx])
    np.save(os.path.join(OUTPUT_DIR, 'val_y.npy'), Y[train_idx:val_idx])

    # 保存 Test
    np.save(os.path.join(OUTPUT_DIR, 'test_sparse.npy'), X_sparse[val_idx:])
    np.save(os.path.join(OUTPUT_DIR, 'test_seq.npy'), seqs[val_idx:])
    np.save(os.path.join(OUTPUT_DIR, 'test_y.npy'), Y[val_idx:])

    # 保存 Meta Info (加入 feature_dims)
    meta = {
        'user_num': user_num,
        'item_num': item_num,
        'sparse_cols': sparse_cols,
        'target_cols': target_names,
        'max_seq_len': MAX_SEQ_LEN,
        'feature_dims': feature_dims
    }

    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Done! 数据已保存在:", OUTPUT_DIR)
    print("Meta info 中的 feature_dims:", feature_dims)


if __name__ == '__main__':
    process()