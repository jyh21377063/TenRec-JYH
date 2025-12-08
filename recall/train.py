import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
import time
import json
import logging
import pickle
import shutil
from datetime import datetime

from dataset import SBRDataset
from model.dssm.dssm import TwoTowerModel
from evaluation import Evaluator
from loss import InfoNCELoss

# ==========================
# 1. 配置区域 (Hyperparameters)
# ==========================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 4096,
    'lr': 1e-3,
    'epochs': 50,  # 设大一点，依靠 Early Stopping 停下来
    'patience': 3,  # 容忍多少个 epoch 指标不提升
    'data_path': '../../data/sbr_data_1208.pkl',
    'exp_dir': '../../experiments',  # 实验结果根目录
    'main_metric': 'Recall@20',  # 用于判断模型好坏的主指标
    'weight_decay': 1e-5,  # 增加一点正则化
    'tau': 0.1  # 温度系数
}


# ==========================
# 2. 工具类：实验管理器
# ==========================
class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.metric_keys = [
            'Recall@10', 'Recall@20', 'Recall@50',
            'NDCG@10', 'NDCG@20', 'NDCG@50',
            'MRR',
            'Avg_Popularity@10', 'Cat_Diversity@10',
            'coverage@10'
        ]

        # 生成带时间戳的实验路径: experiments/EXP_20231208_123000
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(config['exp_dir'], f"EXP_{timestamp}")
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        self.csv_path = os.path.join(self.run_dir, 'metrics.csv')

        # 创建目录
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))

        # 初始化 Logger
        self._setup_logger()

        # 备份 Config
        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # 初始化 csv 表头
        with open(self.csv_path, 'w') as f:
            header = "epoch,loss,phase," + ",".join(self.metric_keys) + "\n"  # 增加 phase 字段区分 val/test
            f.write(header)

        self.logger.info(f"Experiment started. Saving to: {self.run_dir}")

    def _setup_logger(self):
        self.logger = logging.getLogger('SBR_Train')
        self.logger.setLevel(logging.INFO)

        # File Handler
        fh = logging.FileHandler(os.path.join(self.run_dir, 'train_log.txt'))
        fh.setLevel(logging.INFO)

        # Console Handler
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

    def save_model(self, model, optimizer, epoch, metrics, is_best=False):
        """保存模型状态、优化器状态和当前指标"""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 保存 last_model.pth
        last_path = os.path.join(self.ckpt_dir, 'last_model.pth')
        torch.save(state, last_path)

        # 如果是最佳模型，额外保存一份 best_model.pth
        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_model.pth')
            shutil.copyfile(last_path, best_path)
            self.logger.info(
                f" Best model updated at Epoch {epoch} ({self.config['main_metric']}: {metrics[self.config['main_metric']]:.4f})")

    def save_meta(self, meta_info):
        """保存 meta 信息，推理时加载模型必须用到"""
        with open(os.path.join(self.run_dir, 'meta_data.pkl'), 'wb') as f:
            pickle.dump(meta_info, f)

    def log_csv(self, epoch, loss, results, phase='val'):
        with open(self.csv_path, 'a') as f:
            metric_values = [f"{results.get(k, 0.0):.4f}" for k in self.metric_keys]
            line = f"{epoch},{loss:.4f},{phase}," + ",".join(metric_values) + "\n"
            f.write(line)


# ==========================
# 3. 工具类：早停 (Early Stopping)
# ==========================
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric_val = -float('inf')

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.best_metric_val = score
            return True  # Is Best
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_metric_val = score
            self.counter = 0
            return True  # Is Best


# ==========================
# 4. 主训练逻辑
# ==========================
def train():
    # 初始化实验管理器
    exp = ExperimentManager(CONFIG)
    device = torch.device(CONFIG['device'])

    # 1. 加载数据
    exp.log_text("Loading datasets...")
    with open(CONFIG['data_path'], 'rb') as f:
        all_data = pickle.load(f)
    train_ds = SBRDataset(data=all_data, mode='train')
    val_ds = SBRDataset(data=all_data, mode='val')
    test_ds = SBRDataset(data=all_data, mode='test')
    meta = train_ds.get_meta()
    exp.save_meta(meta)

    def extract_ids(mode_data):
        # 只保留 user_id 和 item_id 用于计算 Recall/NDCG
        return {
            'user_id': mode_data['user_id'],
            'item_id': mode_data['item_id']
        }

    full_dataset = {
        'train': extract_ids(all_data['train']),
        'test': extract_ids(all_data['test']),
        'val': extract_ids(all_data['val']),
        'meta': meta
    }

    del all_data
    import gc
    gc.collect()
    exp.log_text("Raw data cleared. Memory optimized.")

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 2. 初始化模型
    model = TwoTowerModel(meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = InfoNCELoss(temperature=CONFIG['tau'])

    evaluator_val = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])
    evaluator_test = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])
    early_stopper = EarlyStopping(patience=CONFIG['patience'])

    exp.log_text(f"Model initialized on {device}. Start Training...")

    # 全局 Step 计数器
    global_step = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        steps_in_epoch = 0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Forward
            u_emb, i_emb = model(batch)

            # In-Batch Negatives
            # (B, D) x (D, B) -> (B, B)
            loss = criterion(u_emb, i_emb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            loss_val = loss.item()
            total_loss += loss_val
            steps_in_epoch += 1
            global_step += 1

            # Tensorboard: 记录每一步的 Loss
            exp.log_scalar('Train/Step_Loss', loss_val, global_step)
            pbar.set_postfix({'loss': loss_val})

        # Epoch 结束统计
        avg_loss = total_loss / steps_in_epoch
        exp.log_scalar('Train/Avg_Loss', avg_loss, epoch)
        exp.log_text(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # ==========================
        # Evaluation & Checkpointing
        # ==========================
        exp.log_text(f"Running Validation for Epoch {epoch}...")
        # 使用 evaluator_val 在 val_loader 上评估
        val_results = evaluator_val.evaluate(val_loader)

        # 记录所有指标到 TensorBoard
        for k, v in val_results.items():
            exp.log_scalar(f"Eval/{k}", v, epoch)

        # 打印日志
        res_str = f"[Val] Epoch {epoch}: " + ", ".join(
            [f"{k}:{v:.4f}" for k, v in val_results.items() if 'Recall' in k or 'NDCG' in k])
        exp.log_text(res_str)

        # 写入 CSV
        exp.log_csv(epoch, avg_loss, val_results, phase='val')

        # Early Stopping Check (基于验证集)
        current_score = val_results[CONFIG['main_metric']]
        is_best = early_stopper(current_score)

        exp.save_model(model, optimizer, epoch, val_results, is_best)

        if early_stopper.early_stop:
            exp.log_text(f" Early stopping triggered! No improvement for {CONFIG['patience']} epochs.")
            exp.log_text(f"Best {CONFIG['main_metric']}: {early_stopper.best_metric_val:.4f}")
            break

    # ==========================
    # Testing Phase
    # ==========================
    exp.log_text("Training loop finished. Starting Final Test Evaluation...")

    # 1. 加载最佳模型权重 (Best Model)
    best_model_path = os.path.join(exp.ckpt_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        exp.log_text(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        exp.log_text(" No best model found, using last epoch model instead.")

    # 2. 在测试集上评估
    test_results = evaluator_test.evaluate(test_loader)

    # 3. 记录 Test 结果
    exp.log_text("=" * 30)
    exp.log_text("FINAL TEST RESULTS (Best Model)")
    exp.log_text("=" * 30)

    test_res_str = ", ".join([f"{k}:{v:.4f}" for k, v in test_results.items()])
    exp.log_text(test_res_str)

    # 将测试结果也写入 CSV，epoch 标记为 999 或 'test'
    exp.log_csv('Test', 0.0, test_results, phase='test')

    exp.writer.close()
    exp.log_text("Experiment Finished Successfully.")


if __name__ == "__main__":
    train()