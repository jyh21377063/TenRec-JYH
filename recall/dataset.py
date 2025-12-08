import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
import gc

class SBRDataset(Dataset):
    def __init__(self, data_path=None, data=None, mode='train', hard_neg_path=None):
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

        if mode == 'train' and hard_neg_path:
            max_item = self.meta['num_items']
            self.item_cat_lookup = torch.zeros(max_item, dtype=torch.long)
            self.item_pop_lookup = torch.zeros(max_item, 1, dtype=torch.float32)

            print("Building Global Item Lookup Table...")
            # 方案：我们需要访问所有数据来填满这个表。
            # 如果 data 包含了 train/val/test，我们遍历一遍来覆盖
            # 注意：如果 data_path 是分开的文件，这里需要额外处理。
            # 假设传入的 'full_data' 字典里包含了 'train', 'val', 'test'
            if data is not None:
                source_splits = ['train', 'val', 'test']
                for split in source_splits:
                    if split in data:
                        split_ids = data[split]['item_id']  # numpy array
                        split_cats = data[split]['video_category']
                        split_pops = data[split]['item_pop_norm']

                        # 转换为 tensor 并填充
                        # 使用高级索引：多次覆盖没关系，特征是静态的
                        t_ids = torch.from_numpy(split_ids.astype(np.int64))
                        t_cats = torch.from_numpy(split_cats.astype(np.int64))
                        t_pops = torch.from_numpy(split_pops.astype(np.float32))

                        if t_pops.dim() == 1: t_pops = t_pops.unsqueeze(1)

                        self.item_cat_lookup[t_ids] = t_cats
                        self.item_pop_lookup[t_ids] = t_pops
            else:
                print("Warning: Only constructing lookup from current split. Hidden items may have zero features.")
                # (保留你原来的代码作为 fallback)
                all_ids = self.tensors['item_id'].long()
                self.item_cat_lookup[all_ids] = self.tensors['video_category']
                self.item_pop_lookup[all_ids] = self.tensors['item_pop_norm']

            # ==========================================
            # 加载 Hard Negative ID (N, K)
            # ==========================================
            self.hard_negs = None
            if hard_neg_path and os.path.exists(hard_neg_path):
                print(f"Loading Hard Negatives from {hard_neg_path}...")
                self.hard_negs = np.load(hard_neg_path)  # 现在的形状是 (N, 10)
            else:
                if hard_neg_path: print("Warning: Hard negative path provided but file not found.")

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

        if hasattr(self, 'hard_negs') and self.hard_negs is not None:
            hn_ids = self.hard_negs[idx]  # 这里拿到的是 (K, ) 的 numpy 数组

            # 转换为 Tensor (K, )
            hn_id_tensor = torch.tensor(hn_ids, dtype=torch.long)

            # 存入 batch
            batch['hn_item_id'] = hn_id_tensor
            batch['hn_video_category'] = self.item_cat_lookup[hn_id_tensor]
            batch['hn_item_pop_norm'] = self.item_pop_lookup[hn_id_tensor]
        return batch

    def get_meta(self):
        return self.meta


# class SBRDatasetMMAP(Dataset):
#     def __init__(self, data_dir, mode='train'):
#         """
#         Args:
#             data_dir: 转换后的文件夹路径
#             mode: 'train', 'val', or 'test'
#         """
#         self.data_dir = data_dir
#         self.mode = mode
#
#         # 1. 加载 Meta
#         with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
#             self.meta = pickle.load(f)
#
#         self.length = 0
#
#         # 【关键修改 1】这里只存文件路径，不存打开的 mmap 对象
#         self.file_paths = {}
#         # 这是一个占位符，真正的数据将在子进程中加载
#         self.mmap_data = None
#
#         files = os.listdir(data_dir)
#         prefix = f"{mode}_"
#
#         for fname in files:
#             if not fname.startswith(prefix) or not fname.endswith('.npy'):
#                 continue
#
#             key = fname[len(prefix):-4]
#             file_path = os.path.join(data_dir, fname)
#
#             # 存路径
#             self.file_paths[key] = file_path
#
#             # 为了获取长度，我们需要临时打开一次，读取后立即释放
#             if key == 'item_id':
#                 temp_mmap = np.load(file_path, mmap_mode='r')
#                 self.length = len(temp_mmap)
#                 del temp_mmap  # 立即删除，防止被 Pickle
#
#         print(f"[{mode}] Dataset initialized. Length: {self.length} (Lazy Loading Mode)")
#
#     def _init_mmap(self):
#         """
#         子进程初始化函数：真正打开 mmap 的地方
#         """
#         if self.mmap_data is None:
#             self.mmap_data = {}
#             for key, path in self.file_paths.items():
#                 self.mmap_data[key] = np.load(path, mmap_mode='r')
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         # 【关键修改 2】如果还没有加载数据（说明是在新进程里第一次运行），则加载
#         if self.mmap_data is None:
#             self._init_mmap()
#
#         batch = {}
#         for key, mmap_arr in self.mmap_data.items():
#             val = mmap_arr[idx]
#
#             if np.issubdtype(val.dtype, np.integer):
#                 batch[key] = torch.tensor(val, dtype=torch.long)
#             elif np.issubdtype(val.dtype, np.floating):
#                 batch[key] = torch.tensor(val, dtype=torch.float32)
#             else:
#                 batch[key] = torch.tensor(val)
#
#             if key in ['user_activity_norm', 'item_pop_norm']:
#                 if batch[key].dim() == 0:
#                     batch[key] = batch[key].unsqueeze(0)
#
#         return batch
#
#     def get_meta(self):
#         return self.meta