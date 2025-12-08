import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import gc


class SBRDatasetMMAP(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        Args:
            data_dir: 转换后的文件夹路径
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.mode = mode

        # 1. 加载 Meta
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            self.meta = pickle.load(f)

        self.length = 0

        # 【关键修改 1】这里只存文件路径，不存打开的 mmap 对象
        self.file_paths = {}
        # 这是一个占位符，真正的数据将在子进程中加载
        self.mmap_data = None

        files = os.listdir(data_dir)
        prefix = f"{mode}_"

        for fname in files:
            if not fname.startswith(prefix) or not fname.endswith('.npy'):
                continue

            key = fname[len(prefix):-4]
            file_path = os.path.join(data_dir, fname)

            # 存路径
            self.file_paths[key] = file_path

            # 为了获取长度，我们需要临时打开一次，读取后立即释放
            if key == 'item_id':
                temp_mmap = np.load(file_path, mmap_mode='r')
                self.length = len(temp_mmap)
                del temp_mmap  # 立即删除，防止被 Pickle

        print(f"[{mode}] Dataset initialized. Length: {self.length} (Lazy Loading Mode)")

    def _init_mmap(self):
        """
        子进程初始化函数：真正打开 mmap 的地方
        """
        if self.mmap_data is None:
            self.mmap_data = {}
            for key, path in self.file_paths.items():
                self.mmap_data[key] = np.load(path, mmap_mode='r')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 【关键修改 2】如果还没有加载数据（说明是在新进程里第一次运行），则加载
        if self.mmap_data is None:
            self._init_mmap()

        batch = {}
        for key, mmap_arr in self.mmap_data.items():
            val = mmap_arr[idx]

            if np.issubdtype(val.dtype, np.integer):
                batch[key] = torch.tensor(val, dtype=torch.long)
            elif np.issubdtype(val.dtype, np.floating):
                batch[key] = torch.tensor(val, dtype=torch.float32)
            else:
                batch[key] = torch.tensor(val)

            if key in ['user_activity_norm', 'item_pop_norm']:
                if batch[key].dim() == 0:
                    batch[key] = batch[key].unsqueeze(0)

        return batch

    def get_meta(self):
        return self.meta


class SBRDataset(Dataset):
    def __init__(self, data_path=None, data=None, mode='train'):
        """
        Args:
            data_path: 你的 .pkl 文件路径
            mode: 'train', 'val', or 'test'
        """
        if data is not None:
            full_data = data
        else:
            print(f"Loading {mode} data from {data_path}...")
            with open(data_path, 'rb') as f:
                full_data = pickle.load(f)

        data_dict = full_data[mode]
        self.meta = full_data['meta']
        self.mode = mode

        # 预先获取数据长度
        self.length = len(data_dict['item_id'])

        # 初始化存储 Tensor 的字典
        self.tensors = {}

        # ==========================================
        # 1. 预处理：一次性将 Numpy 转为 Tensor
        # ==========================================

        # A. 离散特征 (Sparse) -> LongTensor
        # item_id, video_category, gender, age
        sparse_keys = ['item_id', 'video_category', 'gender', 'age']

        for key in sparse_keys:
            if key in data_dict:
                # 转换为 LongTensor (int64)
                # 使用 from_numpy 共享内存，直到原数据被删除
                self.tensors[key] = torch.from_numpy(data_dict[key].astype(np.int64))

        # B. 连续/稠密特征 (Dense) -> FloatTensor
        # user_activity_norm, item_pop_norm
        dense_keys = ['user_activity_norm', 'item_pop_norm']

        for key in dense_keys:
            if key in data_dict:
                arr = data_dict[key].astype(np.float32)
                # 如果是一维数组 (N,)，需要升维成 (N, 1) 方便后续拼接
                if len(arr.shape) == 1:
                    arr = arr[:, np.newaxis]
                self.tensors[key] = torch.from_numpy(arr)

        # C. 序列特征 (Sequence) -> LongTensor
        # item_id_seq, video_category_seq 等
        seq_keys = self.meta['user_tower_seq']

        for key in seq_keys:
            if key in data_dict:
                self.tensors[key] = torch.from_numpy(data_dict[key].astype(np.int64))

        # D. 标签 (Label) - 可选
        if 'click' in data_dict:
            self.tensors['label'] = torch.from_numpy(data_dict['click'].astype(np.float32))

        # self.data = data_dict
        del data_dict

        print(f"[{mode}] Dataset initialized. Memory optimized.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        极速读取：直接从预存的 Tensor 字典中切片
        """
        # 这里的切片操作非常快，且返回的已经是 Tensor，无需再次转换
        batch = {k: v[idx] for k, v in self.tensors.items()}
        return batch

    def get_meta(self):
        return self.meta