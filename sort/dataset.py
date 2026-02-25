import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class MultiTaskDataset(Dataset):
    def __init__(self, sparse_path, y_path, seq_item_path=None, seq_bhv_path=None, use_targets=None, meta_info=None,
                 max_seq_len=20):
        self.use_targets = use_targets
        self.meta_info = meta_info
        self.max_seq_len = max_seq_len  # 新增：允许模型层动态控制序列长度

        print(f"Loading sparse data from {sparse_path}...")
        self.sparse = np.load(sparse_path, mmap_mode='r')

        # 分别加载 item 序列和 behavior 序列
        self.seq_item = None
        self.seq_bhv = None

        if seq_item_path and os.path.exists(seq_item_path):
            self.seq_item = np.load(seq_item_path, mmap_mode='r')
            self.seq_bhv = np.load(seq_bhv_path, mmap_mode='r')
        else:
            print("No sequence data found. Returning dummy sequences.")

        self.y = np.load(y_path, mmap_mode='r')
        self.length = self.sparse.shape[0]

        if self.use_targets and self.meta_info:
            all_target_cols = self.meta_info['target_cols']
            self.target_indices = [all_target_cols.index(t) for t in self.use_targets]
        else:
            self.target_indices = None

    def __getitem__(self, index):
        sparse_data = self.sparse[index]

        # 动态截断：取最后 max_seq_len 个元素 (因为是最新的行为)
        if self.seq_item is not None:
            seq_item_data = self.seq_item[index][-self.max_seq_len:]
            seq_bhv_data = self.seq_bhv[index][-self.max_seq_len:]
        else:
            seq_item_data = np.zeros(self.max_seq_len, dtype=np.int32)
            seq_bhv_data = np.zeros(self.max_seq_len, dtype=np.int32)

        y_data = self.y[index]
        if self.target_indices is not None:
            y_data = y_data[self.target_indices]

        # 返回 4 个 Tensor：稀疏特征, 序列Item, 序列Behavior, 标签
        return (
            torch.from_numpy(sparse_data.copy()).long(),
            torch.from_numpy(seq_item_data.copy()).long(),
            torch.from_numpy(seq_bhv_data.copy()).long(),
            torch.from_numpy(y_data.copy()).float()
        )

    def __len__(self):
        return self.length


class MTLDataManager:
    def __init__(self, config):
        self.cfg = config
        self.data_dir = config.data_dir

        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        import pickle
        with open(meta_path, 'rb') as f:
            self.meta_info = pickle.load(f)

        self.use_targets = self.cfg.use_targets
        # 新增：从 config 中读取序列长度，默认为 20
        self.max_seq_len = getattr(self.cfg, 'max_seq_len', 20)

    def get_dataloader(self, split='train', shuffle=None):
        if shuffle is None:
            shuffle = (split == 'train')

        sparse_path = os.path.join(self.data_dir, f'{split}_sparse.npy')
        seq_item_path = os.path.join(self.data_dir, f'{split}_seq_item.npy')
        seq_bhv_path = os.path.join(self.data_dir, f'{split}_seq_bhv.npy')
        y_path = os.path.join(self.data_dir, f'{split}_y.npy')

        dataset = MultiTaskDataset(
            sparse_path=sparse_path,
            y_path=y_path,
            seq_item_path=seq_item_path,
            seq_bhv_path=seq_bhv_path,
            use_targets=self.use_targets,
            meta_info=self.meta_info,
            max_seq_len=self.max_seq_len  # 传入动态长度
        )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data_num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.data_num_workers > 0)
        )

    def get_feature_info(self):
        return self.meta_info

    def get_num_tasks(self):
        return len(self.use_targets)

    def get_model_feature_dict(self):
        sparse_cols = self.meta_info['sparse_cols']
        feature_dims = self.meta_info.get('feature_dims', {})

        # 获取全局默认的 embedding 维度，若未设置默认为 64
        default_emb_dim = getattr(self.cfg, 'emb_dim', 64)

        # === 在这里手动设置你的自定义特征维度 ===
        # 一般经验法则：dim = 6 * (vocab_size ** 0.25) ，或者凭经验设置
        custom_emb_dims = {
            'gender': 4,  # 3个取值
            'age': 4,  # 7个取值
            'video_category': 32  # 这个忘记统计了，32维应该够了
        }

        feature_dict = {}
        for idx, col in enumerate(sparse_cols):
            vocab_size = feature_dims.get(col, 100)

            # 查找是否配置了特定的维度，否则使用默认维度
            dim = custom_emb_dims.get(col, default_emb_dim)

            # 返回三元组: (词表大小, 该特征专用的 embedding 维度, 在输入中的列索引)
            feature_dict[col] = (vocab_size, dim, idx)

        return feature_dict