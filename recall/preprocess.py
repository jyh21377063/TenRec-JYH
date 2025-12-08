import pickle
import numpy as np
import os
from tqdm import tqdm

# 配置路径
SRC_PATH = '../../data/sbr_data_1208.pkl'
DST_DIR = '../../data/sbr_data_1208_mmap'


def convert_to_mmap():
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        print(f"Created directory: {DST_DIR}")

    print(f"Loading original data from {SRC_PATH} (this uses RAM temporarily)...")
    with open(SRC_PATH, 'rb') as f:
        full_data = pickle.load(f)

    # 1. 保存 Meta 信息 (比较小，还是用 pickle)
    print("Saving metadata...")
    with open(os.path.join(DST_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(full_data['meta'], f)

    # 2. 保存各分区的 Numpy 数组
    modes = ['train', 'val', 'test']

    for mode in modes:
        print(f"Processing {mode} split...")
        data_dict = full_data[mode]

        # 遍历该模式下的所有特征列 (item_id, user_seq, etc.)
        for key, array in tqdm(data_dict.items()):
            # 确保是 numpy array
            if isinstance(array, list):
                array = np.array(array)

            save_path = os.path.join(DST_DIR, f'{mode}_{key}.npy')
            np.save(save_path, array)

    print("Conversion finished! You can now use the new Dataset class.")

    # 释放内存
    del full_data
    import gc
    gc.collect()


if __name__ == "__main__":
    convert_to_mmap()