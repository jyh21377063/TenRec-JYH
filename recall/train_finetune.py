import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import tqdm
import json
import logging
import shutil
from datetime import datetime

from dataset import SBRDataset
from model.dssm.dssm import TwoTowerModel
from evaluation import Evaluator
from train import EarlyStopping
from loss import MixedInfoNCELoss



CONFIG = {
    'device': 'cuda',
    'batch_size': 2048,
    'lr': 1e-4,
    'epochs': 10,
    'patience': 3,
    'data_path': '../../data/sbr_data_1208.pkl',
    'hard_neg_path': '../../experiments/EXP_20251208_151742/data/hard_negatives_train.npy',
    'model_path': '../../experiments/EXP_20251208_151742/checkpoints/best_model.pth',
    'tau': 0.1,
    'main_metric': 'Recall@20',
    'hard_neg_weight': 1.0
}


class FinetuneExperimentManager:
    def __init__(self, config):
        self.config = config
        self.metric_keys = [
            'Recall@10', 'Recall@20', 'Recall@50',
            'NDCG@10', 'NDCG@20', 'NDCG@50',
            'MRR', 'Avg_Popularity@10', 'Cat_Diversity@10', 'coverage@10'
        ]

        base_model_dir = os.path.dirname(config['model_path'])
        self.exp_root = os.path.dirname(base_model_dir)

        self.run_dir = os.path.join(self.exp_root, 'finetune')
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        self.csv_path = os.path.join(self.run_dir, 'finetune_metrics.csv')

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))

        self._setup_logger()

        with open(os.path.join(self.run_dir, 'finetune_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                header = "epoch,loss,phase," + ",".join(self.metric_keys) + "\n"
                f.write(header)

        self.logger.info(f"Finetune Experiment started. Saving to: {self.run_dir}")

    def _setup_logger(self):
        self.logger = logging.getLogger('SBR_Finetune')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(os.path.join(self.run_dir, 'finetune_log.txt'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_text(self, text):
        self.logger.info(text)

    def log_csv(self, epoch, loss, results, phase='val'):
        with open(self.csv_path, 'a') as f:
            metric_values = [f"{results.get(k, 0.0):.4f}" for k in self.metric_keys]
            line = f"{epoch},{loss:.4f},{phase}," + ",".join(metric_values) + "\n"
            f.write(line)

    def save_model(self, model, optimizer, epoch, metrics, is_best=False):
        """保存模型到 finetune/checkpoints"""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        last_path = os.path.join(self.ckpt_dir, 'last_finetuned_model.pth')
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_finetuned_model.pth')
            shutil.copyfile(last_path, best_path)
            self.logger.info(f" Best Finetuned model updated at Epoch {epoch}")



def train_finetune():
    # 1. 初始化管理器
    exp = FinetuneExperimentManager(CONFIG)
    device = CONFIG['device']

    exp.log_text("Loading Data with Hard Negatives...")
    with open(CONFIG['data_path'], 'rb') as f:
        all_data = pickle.load(f)

    # 构造 Evaluator 需要的 dataset 格式
    def extract_ids(mode_data):
        return {'user_id': mode_data['user_id'], 'item_id': mode_data['item_id']}

    # 必须提取 meta，用于初始化模型
    train_ds_temp = SBRDataset(data=all_data, mode='train')
    meta = train_ds_temp.get_meta()

    full_dataset = {
        'train': extract_ids(all_data['train']),
        'test': extract_ids(all_data['test']),
        'val': extract_ids(all_data['val']),
        'meta': meta
    }

    # DataLoader
    train_ds = SBRDataset(data=all_data, mode='train', hard_neg_path=CONFIG['hard_neg_path'])
    val_ds = SBRDataset(data=all_data, mode='val')  # Val 仅用于 Early Stopping
    test_ds = SBRDataset(data=all_data, mode='test')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 2. 加载预训练模型
    exp.log_text(f"Loading Pre-trained Model from {CONFIG['model_path']}...")
    model = TwoTowerModel(meta).to(device)

    # 加载权重
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 准备评估器 & 优化器
    evaluator_val = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])
    evaluator_test = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])

    criterion = MixedInfoNCELoss(
        temperature=CONFIG['tau'],
        hard_neg_weight=CONFIG['hard_neg_weight']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    early_stopper = EarlyStopping(patience=CONFIG['patience'])

    global_step = 0

    # 4. 微调循环
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        steps_in_epoch = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Finetune Epoch {epoch}")

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)

            # --- A. Forward ---
            u_emb = model.forward_user_tower(batch)  # (B, D)
            i_pos_emb = model.forward_item_tower(batch)  # (B, D)

            # --- B. Forward Hard Negatives (关键修改) ---
            # 1. 获取维度信息
            B, K = batch['hn_item_id'].shape  # e.g., 2048, 10

            # 2. 构造 Flatten 后的 Batch
            # 将 (B, K) -> (B*K)
            hn_batch_flat = {
                'item_id': batch['hn_item_id'].view(-1),  # (B*K, )
                'video_category': batch['hn_video_category'].view(-1),  # (B*K, )
                'item_pop_norm': batch['hn_item_pop_norm'].view(-1, 1)  # (B*K, 1)
            }

            # 3. 通过模型
            i_neg_emb_flat = model.forward_item_tower(hn_batch_flat)  # (B*K, D)

            # 4. 还原维度 -> (B, K, D)
            i_neg_emb = i_neg_emb_flat.view(B, K, -1)

            # --- C. Loss Calculation ---
            loss = criterion(u_emb, i_pos_emb, i_neg_emb)

            # --- C. Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- D. Log ---
            loss_val = loss.item()
            total_loss += loss_val
            steps_in_epoch += 1
            global_step += 1

            exp.log_scalar('Finetune/Step_Loss', loss_val, global_step)
            pbar.set_postfix({'loss': loss_val})

        avg_loss = total_loss / steps_in_epoch
        exp.log_scalar('Finetune/Avg_Loss', avg_loss, epoch)
        exp.log_text(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # --- E. Evaluation ---
        exp.log_text(f"Running Validation for Finetune Epoch {epoch}...")
        val_results = evaluator_val.evaluate(val_loader)

        # Log to TensorBoard
        for k, v in val_results.items():
            exp.log_scalar(f"Finetune_Eval/{k}", v, epoch)

        # Log Text & CSV
        res_str = f"[Val] Epoch {epoch}: " + ", ".join(
            [f"{k}:{v:.4f}" for k, v in val_results.items() if 'Recall' in k])
        exp.log_text(res_str)
        exp.log_csv(epoch, avg_loss, val_results, phase='val')

        # --- F. Checkpoint & Early Stopping ---
        current_score = val_results[CONFIG['main_metric']]
        is_best = early_stopper(current_score)

        # 保存模型到 finetune 文件夹
        exp.save_model(model, optimizer, epoch, val_results, is_best)

        if early_stopper.early_stop:
            exp.log_text(f"Early stopping triggered at epoch {epoch}")
            break

    # ==========================
    # 5. Final Test
    # ==========================
    exp.log_text("Finetuning finished. Starting Final Test Evaluation...")

    # 加载微调后的最佳模型
    best_finetuned_path = os.path.join(exp.ckpt_dir, 'best_finetuned_model.pth')
    if os.path.exists(best_finetuned_path):
        exp.log_text(f"Loading best finetuned model from {best_finetuned_path}")
        checkpoint = torch.load(best_finetuned_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluator_test.evaluate(test_loader)

    exp.log_text("=" * 30)
    exp.log_text("FINAL FINETUNE TEST RESULTS")
    exp.log_text("=" * 30)
    test_res_str = ", ".join([f"{k}:{v:.4f}" for k, v in test_results.items()])
    exp.log_text(test_res_str)

    exp.log_csv('Test', 0.0, test_results, phase='test')
    exp.writer.close()
    exp.log_text("Finetune Experiment Finished.")


if __name__ == "__main__":
    train_finetune()