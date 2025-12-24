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
from model.sequence.comirec import MINDModel
from model.dssm.sas_dssm import GatingTwoTowerSASRec
from model.dssm.sas_dssm_simple import ConcatTwoTowerSASRec
from model.dssm.sas_dssm_residual import ResidualSASRec

# ================= 配置 =================
CONFIG = {
    'data_path': '/root/autodl-tmp/data/sbr_data_1208.pkl',
    'model_path': '/root/autodl-tmp/data/4/best_model.pth',
    'output_path': '/root/autodl-tmp/data/4/mixed_negatives_train.npy',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model': 'MIND',

    'batch_size': 4096,
    'num_workers': 4,
    'faiss_threads': 16,

    # IVF 配置
    'nlist': 1000,
    'nprobe': 20,

    # 工业界经验值：简单负样本应占绝大多数，困难负样本只需 1-2 个即可
    # 总共保存 10 个负样本，训练时 1 Positive vs 10 Negatives
    'num_hard_negatives': 2,  # 困难样本数量 (Top-K 中最难的)
    'num_easy_negatives': 8,  # 简单样本数量 (全局随机)

    # 检索时多取一点，防止过滤掉正样本后不够用
    'top_k_retrieve': 50,
}
# 最终保存的每个 User 的负样本总数
CONFIG['total_save_num'] = CONFIG['num_hard_negatives'] + CONFIG['num_easy_negatives']

faiss.omp_set_num_threads(CONFIG['faiss_threads'])


# =======================================

def get_item_embeddings(model, dataset, device):
    """ 计算 Item Embeddings """
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
    num_items = meta['num_items']

    del data
    gc.collect()

    # 2. 加载模型
    if CONFIG['model'] == 'SAS':
        model = SASRecRecallModel(meta, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1).to(device)
    elif CONFIG['model'] == 'MIND':
        model = MINDModel(meta, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1, num_interests=4).to(device)
    elif CONFIG['model'] == 'SASTT':
        model = GatingTwoTowerSASRec(meta).to(device)
    elif CONFIG['model'] == 'SSAS':
        model = ConcatTwoTowerSASRec(meta).to(device)
    elif CONFIG['model'] == 'RSAS':
        model = ResidualSASRec(meta).to(device)
    else:
        model = TwoTowerModel(meta).to(device)

    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 建立 Item 索引
    item_embs = get_item_embeddings(model, dataset, device)
    print(f"Building FAISS IVF Index...")
    d = item_embs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, CONFIG['nlist'], faiss.METRIC_INNER_PRODUCT)
    index.train(item_embs)
    index.add(item_embs)
    index.nprobe = CONFIG['nprobe']
    del item_embs
    gc.collect()

    print(f"Index Built. Start Mixed Strategy Mining...")

    # 4. 流式挖掘 (Streaming Mining)
    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    all_mixed_negs = []

    # 备用随机池，用于快速生成 Simple Negatives
    # 预生成一个大数组，用完再生成，比每次循环生成快
    random_pool_size = 1000000
    random_pool = np.random.randint(1, num_items, size=random_pool_size, dtype=np.int32)
    pool_ptr = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Mining Mixed Negatives"):
            # A. 计算 User Embedding
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            u_emb = model.forward_user_tower(batch)
            u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=-1)

            B = u_emb.shape[0]
            D = u_emb.shape[-1]
            u_emb_flat = u_emb.view(-1, D).cpu().numpy()

            # B. 检索 (Retrieval) - 获取 Hard Candidates
            D_flat, I_flat = index.search(u_emb_flat, CONFIG['top_k_retrieve'])
            I_reshaped = I_flat.reshape(B, -1)

            # 正样本 (用于过滤)
            pos_batch = batch['item_id'].cpu().numpy().reshape(-1, 1)

            # C. 混合策略构建 (Vectorized Processing for Speed)
            batch_res = []

            # 这里虽然用了循环，但只在 Batch 内部，且逻辑简单
            for row_idx in range(B):
                # 1. === Hard Negatives 处理 ===
                # 获取该 User 的检索结果
                candidates = I_reshaped[row_idx]

                # 过滤掉正样本 和 Padding(0)
                # 注意：Hard Mining 最大的风险是把正样本挖回来当负样本训练（False Negative）
                mask = (candidates != pos_batch[row_idx]) & (candidates != 0)
                valid_hard = candidates[mask]

                # 去重 (MIND 模型可能会召回重复的)
                valid_hard = np.unique(valid_hard)

                # 截取 Top-N Hard
                # 如果 valid_hard 不够，先取全部，后面用 random 补
                num_h = min(len(valid_hard), CONFIG['num_hard_negatives'])
                final_hard = valid_hard[:num_h]

                # 2. === Easy Negatives 处理 ===
                # 我们需要填充的总数 = (预设Hard + 预设Easy) - 已有的Hard
                # 这样即使 Hard 不够，也会用 Easy 补齐，保证总数固定
                needed_random = CONFIG['total_save_num'] - num_h

                # 从预生成池中切片
                if pool_ptr + needed_random > random_pool_size:
                    # 池子用完了，重新生成
                    random_pool = np.random.randint(1, num_items, size=random_pool_size, dtype=np.int32)
                    pool_ptr = 0

                random_candidates = random_pool[pool_ptr: pool_ptr + needed_random]
                pool_ptr += needed_random

                # *可选*：检查随机样本是否撞车正样本（概率极低，追求极致可以加）
                # 这里为了速度，且假设物品池 > 10000，撞车概率可忽略
                # if pos_batch[row_idx] in random_candidates: ...

                # 3. === 合并 ===
                # 结果 = [Hard ..., Random ...]
                combined = np.concatenate([final_hard, random_candidates])
                batch_res.append(combined)

            # 存入列表
            all_mixed_negs.append(np.array(batch_res, dtype=np.int32))

    # 5. 保存
    print("Concatenating and Saving...")
    final_negs = np.concatenate(all_mixed_negs, axis=0)

    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    np.save(CONFIG['output_path'], final_negs)

    print(f"Done! Saved to {CONFIG['output_path']}")
    print(f"Shape: {final_negs.shape}")
    print(f"Strategy: {CONFIG['num_hard_negatives']} Hard + {CONFIG['num_easy_negatives']} Easy per sample.")


if __name__ == "__main__":
    main()