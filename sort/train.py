import torch
import torch.nn as nn
import os
import copy
import numpy as np
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from tqdm import tqdm

from config import Config
from dataset import MTLDataManager
from model.sota.mmoe import AdvancedMMOE
from model.sota.cgc import AdvancedCGC
from model.sota.ple import AdvancedPLE
from utils import ExperimentLogger, EarlyStopping
from evaluation import Evaluator


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 初始值为 0 (由于取了指数，实际权重期望从 1 附近开始)
        self.params = nn.Parameter(torch.zeros(num_tasks, requires_grad=True))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            # 公式: loss / (2 * sigma^2) + log(sigma)
            # 学习 log(sigma^2) 以保证数值稳定
            clamped_param = torch.clamp(self.params[i], min=-5.0, max=5.0)
            precision = torch.exp(-clamped_param)
            total_loss += 0.5 * precision * loss + 0.5 * clamped_param
        return total_loss


def train(model_name):
    # 1. 初始化实验记录器
    Config.model_name = model_name
    Config.experiment_name = f"{model_name}"
    logger = ExperimentLogger(Config)

    # 2. 加载数据
    data_manager = MTLDataManager(Config)

    # 获取 DataLoaders
    train_loader = data_manager.get_dataloader('train')
    val_loader = data_manager.get_dataloader('val')
    test_loader = data_manager.get_dataloader('test')

    all_feature_dict = data_manager.get_model_feature_dict()
    item_feature_names = Config.item_feature_names
    user_feature_names = Config.user_feature_names
    user_feature_dict = {}
    item_feature_dict = {}

    for name, value in all_feature_dict.items():
        if name in item_feature_names:
            # 如果是物品特征，放入 item_feature_dict
            item_feature_dict[name] = value
        elif name in user_feature_names:
            # user_feature_dict (包括 user_id, gender, age等)
            user_feature_dict[name] = value

    print(f"User Features: {list(user_feature_dict.keys())}")
    print(f"Item Features: {list(item_feature_dict.keys())}")

    num_tasks = data_manager.get_num_tasks()

    if model_name == 'MMOE':
        model = AdvancedMMOE(
            feature_dict=all_feature_dict,  # 注意: AdvancedMMOE 初始化需要完整的 feature_dict
            max_seq_len=Config.max_seq_len,
            # emb_dim=Config.emb_dim,
            # num_tasks = len(Config.use_targets),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'CGC':
        model = AdvancedCGC(
            feature_dict=all_feature_dict,
            max_seq_len=Config.max_seq_len,
            device=Config.device
        ).to(Config.device)
    elif model_name == 'PLE':
        model = AdvancedPLE(
            feature_dict=all_feature_dict,
            max_seq_len=Config.max_seq_len,
            device=Config.device
        ).to(Config.device)
    else:
        raise NotImplementedError

    criterion = nn.BCEWithLogitsLoss().to(Config.device)

    awl = None
    if getattr(Config, 'loss_weight_mode', 'manual') == 'auto':
        awl = AutomaticWeightedLoss(num_tasks=num_tasks).to(Config.device)
        logger.log("Using Automatic Uncertainty Weighting for Loss.")
    else:
        logger.log(f"Using Manual Weighting for Loss. Weights: {Config.task_weights}")

    # 对 Embedding 层施加特定的 Weight Decay
    embedding_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'embeddings' in name or 'bhv_emb' in name or 'pos_emb' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    # 配置优化器组
    opt_groups = [
        {'params': embedding_params, 'weight_decay': 0.0},  # No L2 penalty for sparse embeddings
        {'params': other_params, 'weight_decay': getattr(Config, 'weight_decay', 1e-6)}
    ]

    # 如果启用了自动加权，将其参数加入优化器，学习率建议设小一点防止震荡
    if awl is not None:
        opt_groups.append({'params': awl.parameters(), 'weight_decay': 0.0, 'lr': Config.lr * 0.1})

    optimizer = torch.optim.Adam(opt_groups, lr=Config.lr)

    # 初始化评估器和早停
    evaluator = Evaluator(data_manager.use_targets, Config.model_name, Config.device)
    early_stopper = EarlyStopping(patience=Config.patience, mode='max')

    logger.log("Start Training...")

    global_step = 0
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        train_pbar = tqdm(enumerate(train_loader),
                          total=len(train_loader),
                          desc=f"Epoch {epoch + 1}/{Config.epochs}")

        for step, batch in train_pbar:
            x_sparse, seq_item, seq_bhv, y = batch

            x_sparse = x_sparse.to(Config.device, non_blocking=True)
            seq_item = seq_item.to(Config.device, non_blocking=True)
            seq_bhv = seq_bhv.to(Config.device, non_blocking=True)
            y = y.to(Config.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x_sparse, seq_item, seq_bhv)  # 形状: [B, num_tasks]

                # 收集所有子任务的 loss
                task_losses = []
                for i in range(num_tasks):
                    # 注意 logits 的切片
                    task_losses.append(criterion(logits[:, i:i + 1], y[:, i:i + 1]))

                if awl is not None:
                    loss = awl(task_losses)
                else:
                    loss = 0
                    for i, t_loss in enumerate(task_losses):
                        loss += t_loss * Config.task_weights[i]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            global_step += 1

            if step % 100 == 0:
                loss_val = loss.item()
                if awl is not None:
                    with torch.no_grad():
                        weights = torch.exp(-awl.params).cpu().numpy()
                    train_pbar.set_postfix({'loss': f"{loss_val:.4f}", 'w': np.round(weights, 2)})
                else:
                    train_pbar.set_postfix({'loss': f"{loss_val:.4f}"})

                logger.log_scalar('Train/Step_Loss', loss_val, global_step)

        train_avg_loss = total_loss / len(train_loader)
        logger.log_scalar('Train/Epoch_Loss', train_avg_loss, epoch)

        # 5. 验证
        val_metrics = evaluator.evaluate(model, val_loader)

        # Tensorboard: 记录验证指标
        for k, v in val_metrics.items():
            logger.log_scalar(f'Val/{k}', v, epoch)

        # 日志打印
        log_str = f"Epoch {epoch + 1} | Train Loss: {train_avg_loss:.4f} | Val AUC: {val_metrics['avg_auc']:.4f}"
        logger.log(log_str)

        detail_str = " | ".join([f"{n}: AUC={val_metrics[f'{n}_auc']:.3f}" for n in Config.use_targets])
        logger.log(f"   {detail_str}")

        # 写入 CSV
        logger.log_csv(epoch, 'val', val_metrics, Config.use_targets)

        # 6. Checkpoints & Early Stopping
        is_best = early_stopper(val_metrics['avg_auc'])
        logger.save_model(model, optimizer, epoch, val_metrics, is_best=is_best)

        if early_stopper.early_stop:
            logger.log("Early stopping triggered!")
            break

    # 7. 测试集最终评估
    logger.log("Loading best model for testing...")
    # 加载 best_model
    checkpoint = torch.load(os.path.join(logger.ckpt_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluator.evaluate(model, test_loader)

    # 记录测试结果
    logger.log(">>> Final Test Results:")
    logger.log(str(test_metrics))

    logger.log_csv(Config.epochs, 'test', test_metrics, Config.use_targets)

    # 关闭资源
    logger.close()


if __name__ == '__main__':
    model_names = ["PLE"]
    for n in model_names:
        train(model_name=n)