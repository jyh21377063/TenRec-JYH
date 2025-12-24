import os
import torch
import torch.nn as nn
import tqdm
import json
import logging
import shutil
import pickle
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import SBRDataset
from evaluation import Evaluator
from train import EarlyStopping
from loss import MixedInfoNCELoss
from model.dssm.dssm import TwoTowerModel
from model.dssm.sas_dssm import GatingTwoTowerSASRec
from model.dssm.sas_dssm_simple import ConcatTwoTowerSASRec
from model.dssm.sas_dssm_residual import ResidualSASRec
from model.sequence.sasrec import SASRecRecallModel
from model.sequence.comirec import MINDModel

# ==========================
# 1. 配置区域 (针对 5090 优化)
# ==========================
CONFIG = {
    'device': 'cuda',
    # RTX 5090 90G 显存优化：
    # 1. 极大 Batch Size 提高吞吐并优化对比学习效果
    # 2. 开启混合精度 (AMP)
    'batch_size': 16384,  # 如果显存还剩很多，可以尝试 32768
    'num_workers': 8,  # 配合高速 CPU
    'lr': 5e-5,  # 微调时学习率通常要比预训练小
    'epochs': 10,
    'patience': 3,
    'data_path': '/root/autodl-tmp/data/sbr_data_1208.pkl',

    # 路径配置
    'hard_neg_path': '/root/autodl-tmp/data/5/hard_negatives_train.npy',
    'model_path': '/root/autodl-tmp/data/5/best_model.pth',

    'tau': 0.1,
    'main_metric': 'Recall@20',
    'hard_neg_weight': 1.0,
    'use_compile': True,  # 是否使用 torch.compile 加速
}


class FinetuneExperimentManager:
    def __init__(self, config):
        self.config = config
        self.metric_keys = [
            'Recall@10', 'Recall@20', 'Recall@50',
            'NDCG@10', 'NDCG@20', 'NDCG@50',
            'MRR'
        ]

        # 自动推导实验目录
        base_model_dir = os.path.dirname(config['model_path'])  # checkpoints
        self.exp_root = os.path.dirname(base_model_dir)  # EXP_xxx

        # 创建 finetune 子目录
        timestamp = time.strftime("%H%M%S")
        self.run_dir = os.path.join(self.exp_root, f'finetune_{timestamp}')
        self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
        self.csv_path = os.path.join(self.run_dir, 'finetune_metrics.csv')

        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tensorboard'))
        self._setup_logger()

        # 保存配置
        with open(os.path.join(self.run_dir, 'finetune_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # 初始化 CSV
        with open(self.csv_path, 'w') as f:
            header = "epoch,loss,phase," + ",".join(self.metric_keys) + "\n"
            f.write(header)

        self.logger.info(f"🚀 Finetune Experiment Started on {config['device']}")
        self.logger.info(f"📂 Saving to: {self.run_dir}")

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
        # 如果模型被 compile 过，state_dict 会有前缀，通常 save 时需要注意，或者直接 save model.state_dict()
        # 这里为了简单直接保存。如果用 DDP 需要 model.module
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        state = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        last_path = os.path.join(self.ckpt_dir, 'last_finetuned.pth')
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, 'best_finetuned.pth')
            shutil.copyfile(last_path, best_path)
            self.logger.info(f"🏆 Best Finetuned model updated at Epoch {epoch}")


def get_model_instance(model_name, meta, device):
    """根据名称工厂化创建模型"""
    if model_name == 'SAS':
        # 这里的参数需要和 train.py 保持一致，最好从 checkpoint config 里读
        return SASRecRecallModel(meta_info=meta, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1).to(device)
    elif model_name == 'MIND':
        return MINDModel(meta_info=meta, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1, num_interests=4).to(
            device)
    elif model_name == 'SASTT':
        return GatingTwoTowerSASRec(meta).to(device)
    elif CONFIG['model'] == 'SSAS':
        return ConcatTwoTowerSASRec(meta).to(device)
    elif CONFIG['model'] == 'RSAS':
        return ResidualSASRec(meta).to(device)
    else:
        # 默认 DSSM / TwoTower
        return TwoTowerModel(meta).to(device)


def train_finetune():
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32，加速 30/40/50 系显卡
    torch.backends.cudnn.benchmark = True  # 加速卷积/矩阵运算

    exp = FinetuneExperimentManager(CONFIG)
    device = torch.device(CONFIG['device'])

    # 1. 加载数据
    exp.log_text("Loading Data...")
    with open(CONFIG['data_path'], 'rb') as f:
        all_data = pickle.load(f)

    # 提取 meta (必须步骤)
    # 临时实例化一个 dataset 来获取 meta，因为 meta 处理逻辑封装在 Dataset 里了
    _temp_ds = SBRDataset(data=all_data, mode='train')
    meta = _temp_ds.get_meta()

    # 构造 Evaluator 用的 dict
    def extract_ids(mode_data):
        return {'user_id': mode_data['user_id'], 'item_id': mode_data['item_id']}

    full_dataset = {
        'train': extract_ids(all_data['train']),
        'test': extract_ids(all_data['test']),
        'val': extract_ids(all_data['val']),
        'meta': meta
    }

    # 2. DataLoader (性能优化关键点)
    train_ds = SBRDataset(data=all_data, mode='train', hard_neg_path=CONFIG['hard_neg_path'])
    val_ds = SBRDataset(data=all_data, mode='val')
    test_ds = SBRDataset(data=all_data, mode='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,  # 锁页内存，加速 CPU->GPU
        persistent_workers=True,  # 避免每个 Epoch 重启 Worker
        prefetch_factor=4  # 预取
    )
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. 智能加载模型
    exp.log_text(f"🔍 Analyzing checkpoint: {CONFIG['model_path']}")
    checkpoint = torch.load(CONFIG['model_path'], map_location=device, weights_only=False)

    # 从 Checkpoint 的 config 中获取模型类型
    saved_config = checkpoint.get('config', {})
    model_type = saved_config.get('model', 'DSSM')  # 默认为 DSSM
    exp.log_text(f"Detected Model Architecture: {model_type}")

    # 实例化对应的模型
    model = get_model_instance(model_type, meta, device)

    # 加载权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        exp.log_text("✅ Model weights loaded successfully.")
    except Exception as e:
        exp.log_text(f"❌ Error loading weights: {e}")
        return

    # 4. 编译模型 (针对 5090 的极端优化)
    if CONFIG['use_compile'] and hasattr(torch, 'compile'):
        exp.log_text("⚡ Compiling model with torch.compile (Mode: max-autotune)...")
        # max-autotune 最快，但编译时间稍长
        model = torch.compile(model, mode='max-autotune')

    # 5. Loss, Optimizer, Scaler
    criterion = MixedInfoNCELoss(
        temperature=CONFIG['tau'],
        hard_neg_weight=CONFIG['hard_neg_weight']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    early_stopper = EarlyStopping(patience=CONFIG['patience'])

    # 混合精度 Scaler
    scaler = GradScaler()

    # 评估器
    evaluator_val = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])  # 验证集只看主要指标加速
    evaluator_test = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])

    global_step = 0

    # 6. 训练循环
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        steps_in_epoch = 0

        pbar = tqdm.tqdm(train_loader, desc=f"Ep {epoch} (BS={CONFIG['batch_size']})", dynamic_ncols=True)

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)  # non_blocking 配合 pin_memory

            optimizer.zero_grad()

            # --- 混合精度上下文 ---
            with autocast(dtype=torch.float16):  # 或者 torch.bfloat16
                # A. 正样本
                u_emb = model.forward_user_tower(batch)
                i_pos_emb = model.forward_item_tower(batch)

                # B. 负样本 (Flatten -> Forward -> Reshape)
                B, K = batch['hn_item_id'].shape

                # 构造 Flatten 后的 Batch (必须高效)
                # 注意：这里假设 hard negative 数据字段都在 batch 里以 'hn_' 开头
                hn_batch_flat = {
                    'item_id': batch['hn_item_id'].view(-1),
                    'video_category': batch['hn_video_category'].view(-1),
                    'item_pop_norm': batch['hn_item_pop_norm'].view(-1, 1)
                }

                # 处理 item tower forward
                i_neg_emb_flat = model.forward_item_tower(hn_batch_flat)
                i_neg_emb = i_neg_emb_flat.view(B, K, -1)

                # C. Loss
                loss = criterion(u_emb, i_pos_emb, i_neg_emb)

            # --- 反向传播 (Scaler) ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            loss_val = loss.item()
            total_loss += loss_val
            steps_in_epoch += 1
            global_step += 1

            if global_step % 10 == 0:
                exp.log_scalar('Finetune/Step_Loss', loss_val, global_step)
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        avg_loss = total_loss / steps_in_epoch
        exp.log_scalar('Finetune/Avg_Loss', avg_loss, epoch)
        exp.log_text(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        # --- Validation ---
        exp.log_text(f"Evaluating Epoch {epoch}...")
        val_results = evaluator_val.evaluate(val_loader)

        # Log
        res_str = f"[Val] Ep {epoch}: Recall@20: {val_results.get('Recall@20', 0):.4f}"
        exp.log_text(res_str)
        exp.log_csv(epoch, avg_loss, val_results)
        exp.save_model(model, optimizer, epoch, val_results, is_best=False)

        # Early Stop
        current_score = val_results[CONFIG['main_metric']]
        if early_stopper(current_score):
            exp.save_model(model, optimizer, epoch, val_results, is_best=True)

        if early_stopper.early_stop:
            exp.log_text("✋ Early stopping triggered.")
            break

    # ==========================
    # Final Test
    # ==========================
    exp.log_text("🏁 Starting Final Test...")
    best_path = os.path.join(exp.ckpt_dir, 'best_finetuned.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, weights_only=False)
        exp.log_text("Loading best checkpoint for testing...")

        # 检查当前模型是否是被 compile 过的
        if hasattr(model, '_orig_mod'):
            # 如果是编译过的模型，把参数加载给内部的原始模型
            model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果没编译，正常加载
            model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluator_test.evaluate(test_loader)

    exp.log_text("=" * 30)
    test_res_str = ", ".join([f"{k}:{v:.4f}" for k, v in test_results.items()])
    exp.log_text(f"TEST RESULTS: {test_res_str}")
    exp.log_csv('Test', 0.0, test_results, phase='test')
    exp.writer.close()


if __name__ == "__main__":
    train_finetune()