import torch
import numpy as np
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
        self.item_meta_map, self.hot_items_set = self._build_meta_and_hot_items()

    def _build_meta_and_hot_items(self):
        """
        构建元数据映射，并根据 PPT 要求，识别出 Top 1% 的热门 Item
        """
        # 1. 统计流行度
        train_items = self.dataset['train']['item_id']
        # bincount 统计每个 ID 出现的次数
        raw_pop_counts = np.bincount(train_items)

        # 避免 log(0)
        log_pop_counts = np.log1p(raw_pop_counts)

        # 2. 识别 Top 1% 热门物品 (用于 PPT 提到的 "去掉 Top 1% 比较召回精度")
        num_items = len(raw_pop_counts)
        sorted_indices = np.argsort(-raw_pop_counts)  # 降序排列
        top_1_percent_count = int(num_items * 0.01)
        hot_items = set(sorted_indices[:top_1_percent_count])

        # 3. Category Map
        item2cat = {}
        for split in ['train', 'val', 'test']:
            ids = self.dataset[split]['item_id']
            if 'video_category' in self.dataset[split]:
                cats = self.dataset[split]['video_category']
                for i, c in zip(ids, cats):
                    item2cat[i] = c
            else:
                for i in ids:
                    item2cat[i] = 0

        meta_map = {
            'pop_score': log_pop_counts,  # 用于计算新颖性 (Novelty)
            'category': item2cat  # 用于计算多样性 (Diversity)
        }

        return meta_map, hot_items

    def compute_all_item_embeddings(self, batch_size=2048):
        """
        预计算所有 Item Embedding
        """
        self.model.eval()
        num_items = self.dataset['meta']['num_items']
        all_item_embs = []
        item_ids = np.arange(0, num_items)
        total_batches = (num_items + batch_size - 1) // batch_size
        pop_map = self.dataset['meta']['item_pop_map']

        with torch.no_grad():
            for i in tqdm(range(total_batches), desc="Encoding All Items"):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_items)
                batch_ids = item_ids[start:end]

                batch_tensor_ids = torch.tensor(batch_ids, dtype=torch.long).to(self.device)
                batch_cats = [self.item_meta_map['category'].get(idx, 0) for idx in batch_ids]
                batch_tensor_cats = torch.tensor(batch_cats, dtype=torch.long).to(self.device)
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

        return torch.cat(all_item_embs, dim=0)

    def evaluate(self, test_loader):
        self.model.eval()
        item_embs = self.compute_all_item_embeddings().to(self.device)

        metrics = defaultdict(list)

        # 记录被推荐过的所有去重物品 ID，用于计算覆盖率
        global_rec_items_set = set()

        item_chunk_size = 10000
        N = item_embs.shape[0]

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Metrics"):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # --- 1. User Embeddings ---
                user_embs = self.model.forward_user_tower(batch)
                user_embs = F.normalize(user_embs, p=2, dim=-1)

                # --- 2. Block TopK Retrieval ---
                batch_topk_vals = []
                batch_topk_idxs = []

                for i in range(0, N, item_chunk_size):
                    end = min(i + item_chunk_size, N)
                    item_chunk = item_embs[i:end]
                    score_chunk = torch.matmul(user_embs, item_chunk.T)

                    if len(score_chunk.shape) == 3:  # Handle Multi-Interest
                        score_chunk, _ = torch.max(score_chunk, dim=1)

                    vals, idxs = torch.topk(score_chunk, k=self.max_k, dim=1)
                    batch_topk_vals.append(vals)
                    batch_topk_idxs.append(idxs + i)

                all_vals = torch.cat(batch_topk_vals, dim=1)
                all_idxs = torch.cat(batch_topk_idxs, dim=1)

                # Refined TopK
                final_vals, final_indices_idx = torch.topk(all_vals, k=self.max_k, dim=1)
                topk_item_ids = torch.gather(all_idxs, 1, final_indices_idx).cpu().numpy()

                # --- 3. Compute Metrics ---
                target_items = batch['item_id'].cpu().numpy()
                self._calculate_batch_metrics(metrics, topk_item_ids, target_items, global_rec_items_set)

        # --- 4. 汇总所有指标 ---
        final_results = {k: np.mean(v) for k, v in metrics.items()}

        # PPT 指标: 召回覆盖率 (Global Coverage)
        total_items = self.dataset['meta']['num_items']
        final_results['Global_Coverage'] = len(global_rec_items_set) / total_items

        return final_results

    def _calculate_batch_metrics(self, metrics, topk_indices, target_items, global_rec_set):
        B = len(target_items)
        pop_map = self.item_meta_map['pop_score']
        cat_map = self.item_meta_map['category']
        pop_map_len = len(pop_map)

        for i in range(B):
            target = target_items[i]  # Ground Truth Item
            pred_list = topk_indices[i]  # Top-K Recommended Items

            # 更新全局覆盖集合
            for pid in pred_list:
                global_rec_set.add(pid)

            # 查找是否命中
            hits = np.where(pred_list == target)[0]

            # 判断目标是否为 "长尾物品" (不在 Top 1% 热门中)
            is_tail_target = target not in self.hot_items_set

            # --- 计算 Recall, NDCG, Precision, F1 ---
            for k in self.k_list:
                # HIT check
                is_hit = 1.0 if len(hits) > 0 and hits[0] < k else 0.0

                # 1. Recall & Hit-Rate (在单目标测试集中，Recall = Hit-Rate)
                metrics[f'Recall@{k}'].append(is_hit)
                metrics[f'Hit-Rate@{k}'].append(is_hit)  # Alias for clarity

                # 2. Precision: hits / k
                # (注意: 单目标场景下 Precision 最大只能是 1/K，但这是标准定义)
                precision = 1.0 / k if is_hit else 0.0
                metrics[f'Precision@{k}'].append(precision)

                # 3. F1 Score: 2 * (P*R) / (P+R)
                if (precision + is_hit) > 0:
                    f1 = 2 * (precision * is_hit) / (precision + is_hit)
                else:
                    f1 = 0.0
                metrics[f'F1@{k}'].append(f1)

                # 4. NDCG
                if is_hit:
                    rank = hits[0]
                    metrics[f'NDCG@{k}'].append(1.0 / np.log2(rank + 2))
                else:
                    metrics[f'NDCG@{k}'].append(0.0)

                # --- PPT 特别要求: 长尾召回能力 (Tail Metrics) ---
                # 只有当 Ground Truth 是长尾物品时，才计入这个指标
                # 这就是 PPT 里 "去掉 top 1% item，再比较召回精度" 的实现
                if is_tail_target:
                    metrics[f'Tail_Recall@{k}'].append(is_hit)
                    metrics[f'Tail_NDCG@{k}'].append(1.0 / np.log2(hits[0] + 2) if is_hit else 0.0)

            # --- MRR ---
            if len(hits) > 0:
                metrics['MRR'].append(1.0 / (hits[0] + 1))
            else:
                metrics['MRR'].append(0.0)

            # --- PPT 指标: 召回新颖性 (Novelty) ---
            # "计算召回 item 的平均流行度...越低新颖度越高"
            # 这里取 Top-10 算平均
            top10_list = pred_list[:10]
            current_pops = []
            for pid in top10_list:
                if pid < pop_map_len:
                    current_pops.append(pop_map[pid])
                else:
                    current_pops.append(0.0)
            metrics['Novelty_AvgPop@10'].append(np.mean(current_pops))

            # --- PPT 指标: 召回多样性 (Diversity) ---
            # "以类目为用户兴趣领域...比较类目宽度"
            cats = set([cat_map.get(pid, 0) for pid in top10_list])
            metrics['Diversity_Cat@10'].append(len(cats) / len(top10_list))