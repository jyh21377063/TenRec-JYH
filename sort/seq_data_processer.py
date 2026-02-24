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

# 序列长度：离线保存最大长度 50
MAX_SEQ_LEN = 50

USE_COLS = ['user_id', 'item_id', 'click', 'like', 'follow', 'share', 'gender', 'age', 'video_category']


def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. 读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE, usecols=lambda c: c in USE_COLS)
    except ValueError:
        print("列名匹配失败，尝试读取所有列再筛选...")
        df = pd.read_csv(INPUT_FILE)
        available_cols = [c for c in USE_COLS if c in df.columns]
        df = df[available_cols]

    print(f"原始数据行数: {len(df)}")
    df.fillna(0, inplace=True)

    print("2. 特征编码 & 维度计算...")
    feature_dims = {}

    # --- A. User ID ---
    lbe_user = LabelEncoder()
    df['user_id'] = lbe_user.fit_transform(df['user_id'])
    feature_dims['user_id'] = df['user_id'].max() + 1

    # --- B. Item ID (0留给padding) ---
    lbe_item = LabelEncoder()
    df['item_id'] = lbe_item.fit_transform(df['item_id']) + 1
    feature_dims['item_id'] = df['item_id'].max() + 1

    # --- C. 其他类别特征 ---
    exclude_cols = ['user_id', 'item_id', 'click', 'like', 'follow', 'share']
    other_sparse_cols = [c for c in df.columns if c not in exclude_cols]

    for col in other_sparse_cols:
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col])
        feature_dims[col] = df[col].max() + 1

    user_num = feature_dims['user_id']
    item_num = feature_dims['item_id']

    # 新增：记录行为类型的维度 (0: padding, 1: click, 2: like, 3: follow, 4: share)
    feature_dims['behavior_type'] = 5
    print(f"User Num: {user_num}, Item Num: {item_num}")

    # ----------------------------------------------------
    # 3. 生成多行为混合序列
    # ----------------------------------------------------
    print(f"3. 生成多行为动态历史序列 (Max Len = {MAX_SEQ_LEN})...")

    user_ids = df['user_id'].values
    item_ids = df['item_id'].values
    clicks = df['click'].values if 'click' in df.columns else np.zeros(len(df))
    likes = df['like'].values if 'like' in df.columns else np.zeros(len(df))
    follows = df['follow'].values if 'follow' in df.columns else np.zeros(len(df))
    shares = df['share'].values if 'share' in df.columns else np.zeros(len(df))

    # 结果容器：分别记录 item 和 对应的 behavior
    seqs_item = np.zeros((len(df), MAX_SEQ_LEN), dtype=np.int32)
    seqs_bhv = np.zeros((len(df), MAX_SEQ_LEN), dtype=np.int32)

    user_history_map = {}

    for i in tqdm(range(len(df)), desc="Generating Multi-Behavior Sequences"):
        u = user_ids[i]
        item = item_ids[i]

        hist = user_history_map.get(u, [])
        hist_len = len(hist)

        if hist_len > 0:
            # 取出最近的 MAX_SEQ_LEN 个历史记录
            recent_hist = hist[-MAX_SEQ_LEN:]
            actual_len = len(recent_hist)

            # 分离 item 和 behavior，使用 pre-padding 填入数组末尾
            seqs_item[i, -actual_len:] = [x[0] for x in recent_hist]
            seqs_bhv[i, -actual_len:] = [x[1] for x in recent_hist]

        # 判断当前交互的最高行为层级 (递进：转发4 > 关注3 > 点赞2 > 点击1)
        bhv_type = 0
        if shares[i] == 1:
            bhv_type = 4
        elif follows[i] == 1:
            bhv_type = 3
        elif likes[i] == 1:
            bhv_type = 2
        elif clicks[i] == 1:
            bhv_type = 1

        # 只要有正向交互，就计入历史序列字典，供下一条样本使用
        if bhv_type > 0:
            if u not in user_history_map:
                user_history_map[u] = []
            user_history_map[u].append((item, bhv_type))

    # ----------------------------------------------------
    # 4. 保存数据
    # ----------------------------------------------------
    print("4. 保存处理后的数据...")

    target_names = ['click', 'like', 'follow', 'share']
    sparse_cols = [c for c in df.columns if c not in target_names]
    X_sparse = df[sparse_cols].values

    for t in target_names:
        if t not in df.columns: df[t] = 0
    Y = df[target_names].values

    total = len(df)
    train_idx = int(total * 0.8)
    val_idx = int(total * 0.9)

    def save_split(split_name, start_idx, end_idx):
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_sparse.npy'), X_sparse[start_idx:end_idx])
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_seq_item.npy'), seqs_item[start_idx:end_idx])
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_seq_bhv.npy'), seqs_bhv[start_idx:end_idx])
        np.save(os.path.join(OUTPUT_DIR, f'{split_name}_y.npy'), Y[start_idx:end_idx])

    save_split('train', 0, train_idx)
    save_split('val', train_idx, val_idx)
    save_split('test', val_idx, total)

    meta = {
        'user_num': user_num,
        'item_num': item_num,
        'sparse_cols': sparse_cols,
        'target_cols': target_names,
        'max_seq_len_offline': MAX_SEQ_LEN,  # 记录离线最大长度
        'feature_dims': feature_dims
    }

    with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Done! 数据已保存在:", OUTPUT_DIR)


if __name__ == '__main__':
    process()