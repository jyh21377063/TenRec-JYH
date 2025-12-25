import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss


class Evaluator:
    def __init__(self, target_names, model_name, device):
        """
        :param target_names: 当前实验使用的目标列名列表，例如 ['click', 'like']
        """
        self.target_names = target_names
        self.model_name = model_name
        self.device = device

    def evaluate(self, model, loader):
        model.eval()
        # 存储每个任务的真实标签和预测概率
        # 维度: [Task1_List, Task2_List, ...]
        y_trues = [[] for _ in range(len(self.target_names))]
        y_preds = [[] for _ in range(len(self.target_names))]

        total_loss = 0.0
        # 注意：ESMM 内部通常自己计算 Loss，这里为了通用，假设 model 返回 logits
        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                # 显式解包 3 个值
                x_sparse, x_seq, y = batch

                # 放到 GPU
                x_sparse = x_sparse.to(self.device)
                x_seq = x_seq.to(self.device)
                y = y.to(self.device)

                outputs = model(x_sparse, x_seq)

                batch_loss = 0

                # --- 遍历每个任务进行处理 ---
                for i, name in enumerate(self.target_names):
                    # 获取当前任务的 Logits 和 Label
                    logits = outputs[i]
                    label = y[:, i:i + 1]  # 原始 label (比如 click 或 like)

                    # === ESMM 特殊逻辑 ===
                    if self.model_name == 'ESMM':
                        # === ESMM 专属逻辑 (一次性处理所有任务) ===
                        logits_ctr = outputs[0]
                        logits_cvr = outputs[1]

                        label_ctr = y[:, 0:1]
                        label_cvr = y[:, 1:2]
                        label_ctcvr = label_ctr * label_cvr

                        # 概率计算
                        pctr = torch.sigmoid(logits_ctr)
                        pcvr = torch.sigmoid(logits_cvr)
                        pctcvr = pctr * pcvr

                        # --- Loss 计算 ---
                        # Task 1: CTR Loss
                        loss_ctr = criterion(logits_ctr, label_ctr)

                        # Task 2: CTCVR Loss (手写 BCE，因为 pctcvr 已经是概率)
                        epsilon = 1e-10
                        loss_ctcvr = - (label_ctcvr * torch.log(pctcvr + epsilon) +
                                        (1 - label_ctcvr) * torch.log(1 - pctcvr + epsilon)).mean()

                        batch_loss += (loss_ctr + loss_ctcvr)

                        # --- 指标记录 ---
                        # Task 1 (Index 0)
                        y_trues[0].extend(label_ctr.cpu().numpy().flatten())
                        y_preds[0].extend(pctr.cpu().numpy().flatten())

                        # Task 2 (Index 1) -> 注意这里存的是 CTCVR
                        y_trues[1].extend(label_ctcvr.cpu().numpy().flatten())
                        y_preds[1].extend(pctcvr.cpu().numpy().flatten())

                    # 2. 其他模型 (MMOE / SharedBottom 等) 走通用循环
                    else:
                        for i, name in enumerate(self.target_names):
                            logits = outputs[i]
                            label = y[:, i:i + 1]

                            batch_loss += criterion(logits, label)

                            y_trues[i].extend(label.cpu().numpy().flatten())
                            y_preds[i].extend(torch.sigmoid(logits).cpu().numpy().flatten())

                total_loss += batch_loss.item()

        avg_loss = total_loss / len(loader)
        metrics = {'loss': avg_loss}
        avg_auc = 0

        # 计算 AUC 和 LogLoss
        for i, name in enumerate(self.target_names):
            try:
                true_vals = np.array(y_trues[i])
                pred_vals = np.array(y_preds[i])

                auc = roc_auc_score(true_vals, pred_vals)
                ll = log_loss(true_vals, pred_vals)
            except ValueError:
                # 防止只有一个类别导致报错
                auc, ll = 0.5, 0.0

            metric_key = name
            if self.model_name == 'ESMM' and i == 1:
                metric_key = f"{name}_ctcvr"

            metrics[f'{metric_key}_auc'] = auc
            metrics[f'{metric_key}_logloss'] = ll
            avg_auc += auc

        metrics['avg_auc'] = avg_auc / len(self.target_names)
        return metrics