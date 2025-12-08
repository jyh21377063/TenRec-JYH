import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, dataset, device, k_list=[10, 20, 50]):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.k_list = k_list
        self.max_k = max(k_list)

        # 预先构建 Item 元数据映射 (用于计算多样性和热度)
        print("Initializing Evaluator: Building Item Metadata Maps...")
        self.item_meta_map = self._build_item_meta_map()

    def _build_item_meta_map(self):
        """
        构建 item_id -> category 和 item_id -> popularity 的映射
        """
        # 1. Pop Score (从 train 统计)
        train_items = self.dataset['train']['item_id']
        # 注意: bincount 长度是 max_id + 1，所以 index 直接对应 item_id
        pop_counts = np.bincount(train_items)
        pop_counts = np.log1p(pop_counts)

        # 2. Category Map
        item2cat = {}
        # 遍历所有数据确保覆盖所有 Item
        for split in ['train', 'val', 'test']:
            ids = self.dataset[split]['item_id']
            if 'video_category' in self.dataset[split]:
                cats = self.dataset[split]['video_category']
                for i, c in zip(ids, cats):
                    item2cat[i] = c
            else:
                for i in ids:
                    item2cat[i] = 0

        return {
            'pop_score': pop_counts,  # Array: index -> pop
            'category': item2cat  # Dict: item_id -> cat
        }

    def compute_all_item_embeddings(self, batch_size=2048):
        """
        预计算所有 Item 的 Embedding (包含 Padding ID=0)
        这样生成的 Tensor 索引就直接等于 Item ID，无需做 +1/-1 的转换
        """
        self.model.eval()
        num_items = self.dataset['meta']['num_items']
        all_item_embs = []

        # 从 0 开始遍历到 num_items (包含 0 作为 padding)
        item_ids = np.arange(0, num_items)
        total_batches = (num_items + batch_size - 1) // batch_size

        # 获取 Meta Map (从 dataset 中读取处理好的 map)
        pop_map = self.dataset['meta']['item_pop_map']

        with torch.no_grad():
            for i in tqdm(range(total_batches), desc="Encoding All Items"):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_items)
                batch_ids = item_ids[start:end]

                # 构造 Batch
                batch_tensor_ids = torch.tensor(batch_ids, dtype=torch.long).to(self.device)

                # 获取对应的 Category
                batch_cats = [self.item_meta_map['category'].get(idx, 0) for idx in batch_ids]
                batch_tensor_cats = torch.tensor(batch_cats, dtype=torch.long).to(self.device)

                # 获取 Pop Norm
                batch_pops = [pop_map.get(idx, 0.0) for idx in batch_ids]
                batch_pop = torch.tensor(batch_pops, dtype=torch.float32).unsqueeze(1).to(self.device)

                batch = {
                    'item_id': batch_tensor_ids,
                    'video_category': batch_tensor_cats,
                    'item_pop_norm': batch_pop
                }

                emb = self.model.forward_item_tower(batch)
                emb = F.normalize(emb, p=2, dim=1)
                all_item_embs.append(emb.cpu())

        # (Num_Items, Embed_Dim) -> Index i is Item ID i
        return torch.cat(all_item_embs, dim=0)

    def evaluate(self, test_loader):
        self.model.eval()

        # 1. 预计算 Item Pool
        # item_embs 的 index 就是 item_id
        item_embs = self.compute_all_item_embeddings().to(self.device)

        metrics = defaultdict(list)
        rec_item_ids = []
        item_chunk_size = 100000
        N = item_embs.shape[0]

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # 2. 计算 User Embeddings
                user_embs = self.model.forward_user_tower(batch)
                user_embs = F.normalize(user_embs, p=2, dim=1)

                # 3. 分块计算 TopK (Block Matrix Multiplication)
                batch_topk_vals = []
                batch_topk_idxs = []

                for i in range(0, N, item_chunk_size):
                    end = min(i + item_chunk_size, N)
                    item_chunk = item_embs[i:end]  # (Chunk_Size, D)

                    # 计算分数 (B, Chunk_Size)
                    score_chunk = torch.matmul(user_embs, item_chunk.T)

                    # 在 score 计算阶段进行 Masking (防止显存爆炸) ---
                    # 这里的 Masking 比较 trick，为了性能，我们通常先取 TopK，
                    # 只有当 Block 很大且 User 历史很长时才需要在 Block 内部 Mask。
                    # 简单起见，这里先取 Block 的 TopK，最后合并后再 Mask 历史。

                    vals, idxs = torch.topk(score_chunk, k=self.max_k, dim=1)

                    batch_topk_vals.append(vals)
                    batch_topk_idxs.append(idxs + i)  # 加上偏移量，还原为全局 Index (即 Item ID)

                # 4. 合并所有 Block 的结果
                all_vals = torch.cat(batch_topk_vals, dim=1)  # (B, Num_Blocks * K)
                all_idxs = torch.cat(batch_topk_idxs, dim=1)  # (B, Num_Blocks * K)

                # 5. 最终 Top-K (Refined)
                # 从合并后的候选集中再次选出 TopK
                final_vals, final_indices_idx = torch.topk(all_vals, k=self.max_k, dim=1)

                # 使用 gather 从 all_idxs 中提取真正的 Item ID
                # 结果: topk_item_ids 就是预测的 item_id，因为 item_embs 的 index 对齐了 id
                topk_item_ids = torch.gather(all_idxs, 1, final_indices_idx).cpu().numpy()

                # --- 删除了原来错误的 scores 变量代码 ---

                # 6. 计算指标
                target_items = batch['item_id'].cpu().numpy()
                self._calculate_batch_metrics(metrics, topk_item_ids, target_items)

                rec_item_ids.extend(topk_item_ids[:, :10].flatten())

        # 汇总结果
        final_results = {k: np.mean(v) for k, v in metrics.items()}

        # Coverage
        unique_rec_items = len(set(rec_item_ids))
        total_items = self.dataset['meta']['num_items']
        final_results['coverage@10'] = unique_rec_items / total_items

        return final_results

    def _calculate_batch_metrics(self, metrics, topk_indices, target_items):
        B = len(target_items)
        pop_map = self.item_meta_map['pop_score']
        cat_map = self.item_meta_map['category']
        pop_map_len = len(pop_map)

        for i in range(B):
            target = target_items[i]
            pred_list = topk_indices[i]

            hits = np.where(pred_list == target)[0]

            for k in self.k_list:
                is_hit = 1.0 if len(hits) > 0 and hits[0] < k else 0.0
                metrics[f'Recall@{k}'].append(is_hit)
                if is_hit:
                    rank = hits[0]
                    metrics[f'NDCG@{k}'].append(1.0 / np.log2(rank + 2))
                else:
                    metrics[f'NDCG@{k}'].append(0.0)

            if len(hits) > 0:
                metrics['MRR'].append(1.0 / (hits[0] + 1))
            else:
                metrics['MRR'].append(0.0)

            # Business Metrics
            top10_list = pred_list[:10]

            # 安全读取 Popularity (处理越界风险)
            current_pops = []
            for pid in top10_list:
                if pid < pop_map_len:
                    current_pops.append(pop_map[pid])
                else:
                    current_pops.append(0.0)
            metrics['Avg_Popularity@10'].append(np.mean(current_pops))

            cats = set([cat_map.get(pid, 0) for pid in top10_list])
            metrics['Cat_Diversity@10'].append(len(cats) / len(top10_list))