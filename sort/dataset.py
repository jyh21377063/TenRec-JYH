import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class MultiTaskDataset(Dataset):
    def __init__(self, sparse_path, y_path, seq_path=None, use_targets=None, meta_info=None):
        """
        :param sparse_path: 稀疏特征路径 (User, Item, Gender...)
        :param y_path: 标签路径 (Click, Like...)
        :param seq_path: 序列特征路径 (Item_id_list...), 可选
        """
        self.use_targets = use_targets
        self.meta_info = meta_info

        # 1. 加载稀疏特征
        print(f"Loading sparse data from {sparse_path}...")
        self.sparse = np.load(sparse_path, mmap_mode='r')

        # 2. 加载序列特征 (如果存在)
        self.seq = None
        if seq_path and os.path.exists(seq_path):
            print(f"Loading sequence data from {seq_path}...")
            self.seq = np.load(seq_path, mmap_mode='r')
        else:
            print("No sequence data found or provided. Returning dummy sequences.")

        # 3. 加载标签
        self.y = np.load(y_path, mmap_mode='r')

        self.length = self.sparse.shape[0]

        # 处理多任务标签索引
        if self.use_targets and self.meta_info:
            all_target_cols = self.meta_info['target_cols']
            self.target_indices = [all_target_cols.index(t) for t in self.use_targets]
        else:
            self.target_indices = None

    def __getitem__(self, index):
        # 1. 获取稀疏特征
        sparse_data = self.sparse[index]

        # 2. 获取序列特征 (如果没有，返回一个全0的占位符，防止报错)
        if self.seq is not None:
            seq_data = self.seq[index]
        else:
            seq_data = np.zeros(1, dtype=np.int32)  # Dummy

        # 3. 获取标签
        y_data = self.y[index]
        if self.target_indices is not None:
            y_data = y_data[self.target_indices]

        # 注意：mmap 需要 copy() 才能转为 Tensor
        return (
            torch.from_numpy(sparse_data.copy()).long(),
            torch.from_numpy(seq_data.copy()).long(),
            torch.from_numpy(y_data.copy()).float()
        )

    def __len__(self):
        return self.length


class MTLDataManager:
    def __init__(self, config):
        self.cfg = config
        self.data_dir = config.data_dir  # 确保这指向 rebuild_seq_from_1m.py 的输出目录

        # 加载元数据
        meta_path = os.path.join(self.data_dir, 'meta_info.pkl')  # 注意名字可能变了
        # 为了兼容 rebuild 脚本生成的 meta.pkl
        if not os.path.exists(meta_path):
            meta_path = os.path.join(self.data_dir, 'meta.pkl')

        import pickle
        with open(meta_path, 'rb') as f:
            self.meta_info = pickle.load(f)

        self.use_targets = self.cfg.use_targets

    def get_dataloader(self, split='train', shuffle=None):
        if shuffle is None:
            # 训练集需要 shuffle (虽然数据是按时间切分的，但在 Batch 内部打乱有助于梯度下降)
            # 测试集不需要 shuffle
            shuffle = (split == 'train')

        # 构造文件名 (对应 rebuild_seq_from_1m.py 的保存命名)
        sparse_path = os.path.join(self.data_dir, f'{split}_sparse.npy')
        seq_path = os.path.join(self.data_dir, f'{split}_seq.npy')
        y_path = os.path.join(self.data_dir, f'{split}_y.npy')

        if not os.path.exists(sparse_path):
            raise FileNotFoundError(f"Data not found: {sparse_path}")

        dataset = MultiTaskDataset(
            sparse_path=sparse_path,
            y_path=y_path,
            seq_path=seq_path,  # 传入序列路径
            use_targets=self.use_targets,
            meta_info=self.meta_info
        )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data_num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.data_num_workers > 0)
        )

    # 获取维度信息传给模型 Embedding 层
    def get_feature_info(self):
        # 根据 rebuild 脚本的 meta.pkl 返回对应信息
        # 例如: return {'user_num': 100, 'item_num': 5000...}
        return self.meta_info

    def get_num_tasks(self):
        return len(self.use_targets)

    def get_model_feature_dict(self):
        # 1. 获取列名顺序
        sparse_cols = self.meta_info['sparse_cols']
        # 2. 获取计算好的维度字典
        feature_dims = self.meta_info.get('feature_dims', {})

        feature_dict = {}
        for idx, col in enumerate(sparse_cols):
            vocab_size = feature_dims.get(col, 100)
            feature_dict[col] = (vocab_size, idx)

        return feature_dict