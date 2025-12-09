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
from model.sequence.sasrec import SASRecRecallModel

# ================= 配置 =================
CONFIG = {
    'data_path': '/root/autodl-tmp/data/sbr_data_1208.pkl',
    'model_path': '/root/autodl-tmp/data/1/best_model.pth',
    'output_path': '/root/autodl-tmp/data/1/hard_negatives_train.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': 'SAS',

    # Batch Size 稍微调小一点，保证稳定性
    'batch_size': 4096,
    'top_k': 50,
    'top_k_save': 10,

    # 降低 worker 数量，防止 DataLoader 吃光内存
    'num_workers': 4,
    'faiss_threads': 16,  # 检索线程可以高

    # IVF 配置
    'nlist': 1000,
    'nprobe': 20
}

faiss.omp_set_num_threads(CONFIG['faiss_threads'])


# =======================================

def get_item_embeddings(model, dataset, device):
    """
    计算 Item Embeddings (这个量级通常不大，可以存内存)
    """
    print("Generating Item Embeddings...")
    dataset_len = dataset.meta['num_items']

    item_ids = dataset.tensors['item_id'].numpy()
    cats = dataset.tensors['video_category'].numpy()
    pops = dataset.tensors['item_pop_norm'].numpy()

    unique_items = torch.arange(dataset_len, dtype=torch.long)
    unique_cats = torch.zeros(dataset_len, dtype=torch.long)
    unique_pops = torch.zeros(dataset_len, 1, dtype=torch.float32)

    unique_cats[item_ids] = torch.tensor(cats, dtype=torch.long)
    unique_pops[item_ids] = torch.tensor(pops, dtype=torch.float32)

    loader = DataLoader(
        torch.utils.data.TensorDataset(unique_items, unique_cats, unique_pops),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )

    embs = []
    with torch.no_grad():
        for b_ids, b_cats, b_pops in tqdm(loader, desc="Encoding Items"):
            b_ids = b_ids.to(device)
            dummy_batch = {
                'item_id': b_ids,
                'video_category': b_cats.to(device),
                'item_pop_norm': b_pops.to(device)
            }
            emb = model.forward_item_tower(dummy_batch)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.append(emb.cpu().numpy())

    return np.concatenate(embs, axis=0)


def main():
    device = CONFIG['device']

    # 1. 加载数据
    print("Loading Data...")
    with open(CONFIG['data_path'], 'rb') as f:
        data = pickle.load(f)
    dataset = SBRDataset(data=data, mode='train')
    meta = dataset.get_meta()

    # 显式清理，防止 data 和 dataset 双重占用
    del data
    gc.collect()

    # 2. 加载模型
    if CONFIG['model'] == 'SAS':
        model = SASRecRecallModel(
            meta_info=meta,
            embed_dim=64,  # 可以根据需要调整
            num_layers=2,  # Transformer 层数，2层通常对召回足够
            num_heads=2,
            dropout=0.1
        ).to(device)
    else:
        model = TwoTowerModel(meta).to(device)
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 优先处理 Item (建立索引)
    item_embs = get_item_embeddings(model, dataset, device)

    print(f"Building FAISS IVF Index (nlist={CONFIG['nlist']})...")
    d = item_embs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, CONFIG['nlist'], faiss.METRIC_INNER_PRODUCT)

    # 训练并添加
    # 这里的 item_embs 是必须要占内存的，但它只有一份
    index.train(item_embs)
    index.add(item_embs)
    index.nprobe = CONFIG['nprobe']

    # 关键：Item 已经进索引了，Numpy 数组可以删了！省几个G内存
    del item_embs
    gc.collect()

    print(f"Index Built. Start Streaming Mining...")

    # 4. 流式挖掘 (Streaming Mining)
    # 不再一次性算出所有 User，而是边算边搜
    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    all_hard_negs = []
    backup_pool = np.random.randint(1, meta['num_items'], size=(100000,))
    backup_ptr = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Stream Mining"):
            # A. GPU 计算 User 向量
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            u_emb = model.forward_user_tower(batch)
            u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=1)

            # 转回 CPU
            u_emb_np = u_emb.cpu().numpy()
            pos_batch = batch['item_id'].cpu().numpy().reshape(-1, 1)

            # B. 立即检索 (CPU 多线程)
            D, I = index.search(u_emb_np, CONFIG['top_k'])

            # C. 立即处理结果
            mask = (I != pos_batch) & (I != 0)

            batch_res = []
            for row_idx in range(len(I)):
                valid_ids = I[row_idx][mask[row_idx]]

                if len(valid_ids) >= CONFIG['top_k_save']:
                    batch_res.append(valid_ids[:CONFIG['top_k_save']])
                else:
                    needed = CONFIG['top_k_save'] - len(valid_ids)
                    if backup_ptr + needed > len(backup_pool):
                        backup_ptr = 0
                    fill = backup_pool[backup_ptr: backup_ptr + needed]
                    backup_ptr += needed
                    batch_res.append(np.concatenate([valid_ids, fill]))

            # D. 只存轻量级的 int32 结果，User Embedding 在这一步结束后就被释放了
            all_hard_negs.append(np.array(batch_res, dtype=np.int32))

            # 可选：每隔几轮强制 GC 一次，防止内存碎片
            # gc.collect()

    # 5. 保存
    print("Concatenating...")
    final_negs = np.concatenate(all_hard_negs, axis=0)

    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    np.save(CONFIG['output_path'], final_negs)
    print(f"Done! Saved to {CONFIG['output_path']}")


if __name__ == "__main__":
    main()