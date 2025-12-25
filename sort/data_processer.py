import polars as pl
import numpy as np
import pickle
import os
import gc  # 引入垃圾回收
from sklearn.model_selection import train_test_split
from config import Config


def preprocess():
    print(f"Reading CSV from {Config.raw_data_path} with Polars...")

    # 1. 使用 Polars 读取数据 (多线程，速度快)
    df = pl.read_csv(Config.raw_data_path)

    # 2. 填充缺失值 (Polars 语法)
    # sparse features 填 "-1", target 填 0
    print("Filling nulls...")
    df = df.with_columns([
        pl.col(Config.sparse_features).fill_null("-1"),
        pl.col(Config.target_cols).fill_null(0)
    ])

    # 3. Label Encoding & 记录维度
    # Polars 的 Categorical 转换比 sklearn LabelEncoder 快得多
    feature_dims = {}
    print("Encoding features...")

    # 为了保证后续逻辑正确，我们对每一列进行类型转换：String -> Categorical -> Physical (Int)
    # 注意：Physical index 是从 0 开始的整数，相当于 LabelEncoder 的 transform
    exprs = []
    for feat in Config.sparse_features:
        # 这里构建表达式，稍后通过 select 或 with_columns 执行
        # 注意：如果数据量极大，建议逐个处理防止内存峰值
        df = df.with_columns(
            pl.col(feat).cast(pl.String).cast(pl.Categorical).to_physical().alias(feat)
        )
        # 获取唯一值数量 (+1 是为了稳健性，通常最大index+1即为维度，或者直接用 n_unique)
        feature_dims[feat] = df[feat].n_unique()

    # 4. 转换为 Numpy 数组
    print("Converting to Numpy...")
    # 显式指定类型以节省内存 (int32 和 float32 通常足够)
    x_all = df.select(Config.sparse_features).to_numpy().astype(np.int32)
    y_all = df.select(Config.target_cols).to_numpy().astype(np.float32)

    # **关键步骤**：立即删除 Polars DataFrame 并回收内存
    print("Freeing DataFrame memory...")
    del df
    gc.collect()

    # 5. 划分数据集 (8:1:1)
    print("Splitting data...")
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x_all, y_all, test_size=0.2, random_state=Config.seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=Config.seed
    )

    # 再次回收 split 产生的临时变量内存
    del x_all, y_all, x_tmp, y_tmp
    gc.collect()

    # 6. 保存数据
    save_dir = os.path.dirname(Config.processed_data_path)
    if not save_dir: save_dir = '.'  # 防止路径为空
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving splits to {save_dir}...")

    # 保存 Train
    np.save(os.path.join(save_dir, 'train_x.npy'), x_train)
    np.save(os.path.join(save_dir, 'train_y.npy'), y_train)
    print("Train set saved.")
    del x_train, y_train  # 保存完立即释放
    gc.collect()

    # 保存 Val
    np.save(os.path.join(save_dir, 'val_x.npy'), x_val)
    np.save(os.path.join(save_dir, 'val_y.npy'), y_val)
    print("Val set saved.")
    del x_val, y_val
    gc.collect()

    # 保存 Test
    np.save(os.path.join(save_dir, 'test_x.npy'), x_test)
    np.save(os.path.join(save_dir, 'test_y.npy'), y_test)
    print("Test set saved.")
    del x_test, y_test
    gc.collect()

    # 7. 保存元数据 (维度信息)
    meta_info = {
        'feature_dims': feature_dims,
        'sparse_features': Config.sparse_features,
        'target_cols': Config.target_cols
    }

    meta_path = os.path.join(save_dir, 'meta_info.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_info, f)
    print(f"Meta info saved to {meta_path}")

    print("Preprocessing All Done!")


if __name__ == '__main__':
    preprocess()