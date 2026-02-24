import torch
import torch.nn as nn
import os
import copy
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from tqdm import tqdm

from config import Config
from dataset import MTLDataManager
from model.mtl.mmoe import MMOE,MMOE_SEQ,MMOE_DIN,MMOE_DCN_DIN
from model.mtl.esmm import ESMM,ESMM_SEQ,ESMM_DIN,ESMM_DCN_DIN
from model.mtl.ple import PLE_SEQ,PLE_DIN,PLE_DCN_DIN
from utils import ExperimentLogger, EarlyStopping
from evaluation import Evaluator


def train(model_name):
    # 1. 初始化实验记录器
    Config.model_name = model_name
    Config.experiment_name = f"{model_name}_v1"
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

    # model_name = Config.model_name

    if model_name=='MMOE':
        model = MMOE(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_expert=Config.n_expert,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'MMOE_SEQ':
        model = MMOE_SEQ(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_expert=Config.n_expert,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'MMOE_DIN':
        model = MMOE_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_expert=Config.n_expert,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'MMOE_DCN_DIN':
        model = MMOE_DCN_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_expert=Config.n_expert,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'PLE_SEQ':
        # PLE 需要 n_specific_experts 和 n_shared_experts
        model = PLE_SEQ(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_specific_experts=Config.n_specific_experts,
            n_shared_experts=Config.n_shared_experts,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'PLE_DIN':
        # PLE 需要 n_specific_experts 和 n_shared_experts
        model = PLE_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_specific_experts=Config.n_specific_experts,
            n_shared_experts=Config.n_shared_experts,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name == 'PLE_DCN_DIN':
        # PLE 需要 n_specific_experts 和 n_shared_experts
        model = PLE_DCN_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_specific_experts=Config.n_specific_experts,
            n_shared_experts=Config.n_shared_experts,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)
    elif model_name=='ESMM':
        model = ESMM(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
        ).to(Config.device)
    elif model_name=='ESMM_SEQ':
        model = ESMM_SEQ(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
        ).to(Config.device)
    elif model_name=='ESMM_DIN':
        model = ESMM_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
        ).to(Config.device)
    elif model_name=='ESMM_DCN_DIN':
        model = ESMM_DCN_DIN(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            hidden_dim=Config.hidden_dim,
            din_hidden_dim=Config.din_hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
        ).to(Config.device)
    else:
        model = MMOE(
            user_feature_dict=user_feature_dict,
            item_feature_dict=item_feature_dict,
            emb_dim=Config.emb_dim,
            n_expert=Config.n_expert,
            mmoe_hidden_dim=Config.mmoe_hidden_dim,
            hidden_dim=Config.hidden_dim,
            dropouts=Config.dropouts,
            output_size=1,
            num_task=data_manager.get_num_tasks(),
            device=Config.device
        ).to(Config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(Config.device)

    # 初始化评估器和早停
    evaluator = Evaluator(data_manager.use_targets, Config.model_name, Config.device)
    early_stopper = EarlyStopping(patience=Config.patience, mode='max')

    logger.log("Start Training...")

    global_step = 0
    scaler = torch.amp.GradScaler('cuda')
    best_model_weights = None  # 用于在内存中保存最优模型权重

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        train_pbar = tqdm(enumerate(train_loader),
                          total=len(train_loader),
                          desc=f"Epoch {epoch + 1}/{Config.epochs}")

        for step, batch in train_pbar:
            x_sparse, x_seq, y = batch

            x_sparse = x_sparse.to(Config.device, non_blocking=True)
            x_seq = x_seq.to(Config.device, non_blocking=True)
            y = y.to(Config.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits_list = model(x_sparse, x_seq)
                loss = 0

                # === Logic for ESMM ===
                if 'ESMM' in Config.model_name:
                    # Assuming outputs are [logits_ctr, logits_cvr]
                    logits_ctr = logits_list[0]
                    logits_cvr = logits_list[1]

                    # Labels
                    label_ctr = y[:, 0:1]
                    label_cvr = y[:, 1:2]  # Usually 'like' or 'conversion'
                    label_ctcvr = label_ctr * label_cvr

                    # 1. CTR Loss
                    loss_ctr = criterion(logits_ctr, label_ctr)

                    # 2. CTCVR Loss
                    # ESMM: pCTCVR = sigmoid(logits_ctr) * sigmoid(logits_cvr)
                    pctr = torch.sigmoid(logits_ctr)
                    pcvr = torch.sigmoid(logits_cvr)
                    pctcvr = pctr * pcvr

                    # Since we have Probabilities (pctcvr), use BCELoss, NOT WithLogits
                    epsilon = 1e-10
                    loss_ctcvr = - (label_ctcvr * torch.log(pctcvr + epsilon) +
                                    (1 - label_ctcvr) * torch.log(1 - pctcvr + epsilon)).mean()

                    loss = loss_ctr + loss_ctcvr

                else:
                    for i in range(len(logits_list)):
                        loss += criterion(logits_list[i], y[:, i:i + 1])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()

            global_step += 1

            # Tensorboard: 记录 Step Loss
            if step % 100 == 0:
                loss_val = loss.item()
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

        # 详细 LogLoss
        detail_str = " | ".join(
            [f"{n}: AUC={val_metrics[f'{n}_auc']:.3f}" for n in Config.use_targets])
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
    model_names = [
        # "MMOE", "MMOE_SEQ", "MMOE_DIN",
        "MMOE_DCN_DIN",
        "PLE_SEQ", "PLE_DIN", "PLE_DCN_DIN",
        "ESMM_SEQ", "ESMM_DIN","ESMM_DCN_DIN",
    ]
    for n in model_names:
        train(model_name=n)