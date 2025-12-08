import torch
import numpy as np
import faiss
import pickle
import gc
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SBRDataset
from model.dssm.dssm import TwoTowerModel

# ================= 配置 =================
CONFIG = {
    'data_path': '../../data/sbr_data_1208.pkl',
    'model_path': '../../experiments/EXP_20251208_151742/checkpoints/best_model.pth',
    'output_path': '../../experiments/EXP_20251208_151742/hard_negatives_train.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 2048,
    'top_k': 50,
    'top_k_save': 10,
    'num_workers': 2
}


# =======================================

def get_unique_item_features(dataset):
    """
    构建一个 Item 特征表: item_id -> [category, pop_norm]
    因为模型 Forward Item Tower 需要这些特征
    """
    print("Building Item Feature Table...")
    max_item = dataset.meta['num_items']

    # 假设 dataset.tensors 里存的是全量交互数据的特征
    # 我们需要去重，拿到每个 item_id 对应的属性
    # 初始化：category=0, pop=0
    item_cats = torch.zeros(max_item, dtype=torch.long)
    item_pops = torch.zeros(max_item, 1, dtype=torch.float32)

    # 利用 scatter 或者简单的循环去重 (这里用简单的覆盖法，假设同一 item 特征一致)
    item_ids = dataset.tensors['item_id'].numpy()
    cats = dataset.tensors['video_category'].numpy()
    pops = dataset.tensors['item_pop_norm'].numpy()

    # 这种写法简单粗暴，但有效
    item_cats[item_ids] = torch.tensor(cats, dtype=torch.long)
    item_pops[item_ids] = torch.tensor(pops, dtype=torch.float32)

    return item_cats, item_pops


def main():
    device = CONFIG['device']

    # 1. 加载数据
    print("Loading Data...")
    with open(CONFIG['data_path'], 'rb') as f:
        data = pickle.load(f)
    dataset = SBRDataset(data=data, mode='train')
    meta = dataset.get_meta()

    # 内存优化：如果 dataset 内部已经复制了 tensors，
    # 这里的 data 变量本身就没有用了，尝试手动删除引用（具体看 Dataset 实现）
    del data
    gc.collect()

    # 2. 加载模型
    print("Loading Model...")
    model = TwoTowerModel(meta).to(device)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 生成所有 Item 的 Embedding (建立索引库)
    print("Generating Item Embeddings...")
    item_cats, item_pops = get_unique_item_features(dataset)
    all_item_ids = torch.arange(meta['num_items'], dtype=torch.long)

    item_embs_list = []
    batch_size = CONFIG['batch_size']
    num_items = len(all_item_ids)

    with torch.no_grad():
        for i in range(0, num_items, batch_size):
            end = min(i + batch_size, num_items)
            batch_ids = all_item_ids[i:end].to(device)
            batch_cats = item_cats[i:end].to(device)
            batch_pops = item_pops[i:end].to(device)

            # 伪造一个 batch 字典喂给 item tower
            dummy_batch = {
                'item_id': batch_ids,
                'video_category': batch_cats,
                'item_pop_norm': batch_pops
            }
            emb = model.forward_item_tower(dummy_batch)
            # 归一化 (Cosine)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            item_embs_list.append(emb.cpu().numpy())

    # 合并 numpy 数组
    item_embs = np.concatenate(item_embs_list, axis=0)
    del item_embs_list  # 内存优化：删掉 list
    gc.collect()

    # 4. 构建 FAISS 索引
    print("Building FAISS Index...")
    # 维度
    d = item_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_embs)

    # 【关键优化】：Faiss 内部已经存了一份 copy，
    # 这里的 numpy 数组 item_embs 可以删了，释放大量内存！
    print("Freeing item embeddings memory...")
    del item_embs
    gc.collect()

    # 5. 准备磁盘映射 (Memmap) 用于存储结果
    # 这样结果不会占用 RAM，而是直接写入硬盘
    total_samples = len(dataset)
    print(f"Initializing Memmap for {total_samples} samples...")

    # 确保目录存在
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)

    # 创建 memmap 文件
    hard_neg_map = np.memmap(
        CONFIG['output_path'],
        dtype='int32',
        mode='w+',
        shape=(total_samples, CONFIG['top_k_save'])
    )

    # 6. 开始挖掘
    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']  # 降低 worker
    )

    print("Mining Hard Negatives (Streaming to disk)...")
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # User Embedding
            u_emb = model.forward_user_tower(batch)
            u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=1)
            u_emb_np = u_emb.cpu().numpy()

            # 搜索 Top K
            # D: distances, I: indices (item_ids)
            D, I = index.search(u_emb_np, CONFIG['top_k'])

            pos_items = batch['item_id'].cpu().numpy()

            # 当前 batch 的结果缓存
            batch_result = []

            for i in range(len(pos_items)):
                true_item = pos_items[i]
                candidates = I[i]  # Faiss 返回的 Top-K 候选列表

                # 策略：过滤掉 True Item 和 Padding(0)，收集前 10 个
                valid_negs = []
                for cand in candidates:
                    if cand != true_item and cand != 0:
                        valid_negs.append(cand)
                    if len(valid_negs) >= CONFIG['top_k_save']:
                        break

                # 兜底策略：如果过滤完不够 10 个 (极少见)，随机采样补齐
                while len(valid_negs) < CONFIG['top_k_save']:
                    rand_id = np.random.randint(1, meta['num_items'])
                    # 简单去重，尽量不要选到正样本
                    if rand_id != true_item:
                        valid_negs.append(rand_id)

                batch_result.append(valid_negs)

            # 【写入硬盘】
            # 将当前 batch 的结果写入 memmap 对应的位置
            batch_size_curr = len(batch_result)
            hard_neg_map[global_idx: global_idx + batch_size_curr] = np.array(batch_result)

            # 这里的 flush 不是必须每次都调，但为了保险可以定期调
            # hard_neg_map.flush()

            global_idx += batch_size_curr

    # 最后 flush 一下确保写入
    hard_neg_map.flush()
    print(f"Done! Hard negatives saved directly to {CONFIG['output_path']}")

    # 只有 memmap 对象被销毁或者 close 后，文件才完全释放
    del hard_neg_map


if __name__ == "__main__":
    main()