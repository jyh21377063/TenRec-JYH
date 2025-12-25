import os
import torch
import logging
import datetime
import shutil
import json
import csv
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, config):
        # 1. 创建实验目录: experiments/MMOE_v1_20251224_1530
        log_dir = config.log_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{config.experiment_name}_{timestamp}")
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 2. 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))

        # 3. 初始化 Logging
        self.logger = logging.getLogger(config.experiment_name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # 文件处理器
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'train_log.txt'), encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 4. 保存 Config 为 JSON
        self.save_config(config)

        # 5. 初始化 CSV 记录
        self.csv_path = os.path.join(self.exp_dir, 'metrics.csv')
        self.init_csv(config.target_cols)

        self.log(f"Experiment Started: {config.experiment_name}")
        self.log(f"Artifacts saved to: {self.exp_dir}")

    def log(self, message):
        self.logger.info(message)

    def log_scalar(self, tag, value, step):
        """记录到 Tensorboard"""
        self.writer.add_scalar(tag, value, step)

    def save_config(self, config):
        """保存配置到 json"""
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        # 处理 device 对象无法序列化的问题
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])

        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

    def init_csv(self, target_cols):
        """初始化 CSV 表头"""
        headers = ['epoch', 'phase', 'loss', 'avg_auc']
        for target in target_cols:
            headers.extend([f'{target}_auc', f'{target}_logloss'])

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_csv(self, epoch, phase, metrics, target_cols):
        """写入一行 CSV"""
        row = [epoch, phase, metrics.get('loss', 0), metrics.get('avg_auc', 0)]
        for target in target_cols:
            row.extend([
                metrics.get(f'{target}_auc', 0),
                metrics.get(f'{target}_logloss', 0)
            ])

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_model(self, model, optimizer, epoch, metrics, is_best=False):
        """保存 Last 和 Best 模型"""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        # 保存 Last
        last_path = os.path.join(self.ckpt_dir, 'last_model.pth')
        torch.save(state, last_path)

        # 保存 Best
        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_model.pth')
            shutil.copyfile(last_path, best_path)
            self.log(f"*** Best Model Saved (Epoch {epoch}) ***")

    def get_exp_dir(self):
        return self.exp_dir

    def close(self):
        self.writer.close()


# EarlyStopping 需要微调，不再负责保存模型，只负责逻辑判断
class EarlyStopping:
    def __init__(self, patience=3, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric_val = -float('inf')

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            self.best_metric_val = current_score
            return True  # Is Best

        is_improvement = False
        if self.mode == 'max':
            if current_score > self.best_score + self.delta: is_improvement = True
        else:
            if current_score < self.best_score - self.delta: is_improvement = True

        if is_improvement:
            self.best_score = current_score
            self.best_metric_val = current_score
            self.counter = 0
            return True  # Is Best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False