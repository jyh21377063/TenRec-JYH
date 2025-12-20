import torch
import pickle
import os
import json
import numpy as np
from torch.utils.data import DataLoader

from dataset import SBRDataset
from evaluation_offline import Evaluator
from model.dssm.dssm import TwoTowerModel
from model.sequence.sasrec import SASRecRecallModel
from model.sequence.comirec import MINDModel


def load_model_from_checkpoint(checkpoint_path, device):
    """
    加载模型架构和权重
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    metrics = checkpoint['metrics']
    epoch = checkpoint['epoch']

    return checkpoint, config, metrics, epoch


def run_evaluation(data_path, model_ckpt_path, output_file='offline_eval_result.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据 (只加载 Test 和 Meta，节省内存)
    print("Loading Dataset...")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    test_ds = SBRDataset(data=all_data, mode='test')
    meta = test_ds.get_meta()

    # 构造 Evaluator 需要的数据格式
    full_dataset = {
        'train': {'item_id': all_data['train']['item_id']},  # 用于计算流行度
        'test': {'user_id': all_data['test']['user_id'], 'item_id': all_data['test']['item_id']},
        'val': {'item_id': []},  # 离线评估不需要 val
        'meta': meta
    }

    # 清理内存
    del all_data
    import gc
    gc.collect()

    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4)

    # 2. 加载 Checkpoint
    checkpoint, config, saved_metrics, epoch = load_model_from_checkpoint(model_ckpt_path, device)

    print(f"Restoring model: {config['model']} from Epoch {epoch}")

    # 3. 初始化模型结构 (参数必须与训练时一致)
    if config['model'] == 'SAS':
        model = SASRecRecallModel(
            meta_info=meta,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1
        ).to(device)
    elif config['model'] == 'MIND':
        model = MINDModel(
            meta_info=meta,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            num_interests=4
        ).to(device)
    else:
        model = TwoTowerModel(meta).to(device)

    # 4. 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. 运行评估
    print("Starting Offline Evaluation...")
    evaluator = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])

    final_results = evaluator.evaluate(test_loader)

    print("Evaluation Done. Saving results...")

    timestamp = torch.tensor(0).cuda()  # Just dummy
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_str = []
    result_str.append(f"========================================")
    result_str.append(f"Evaluation Report")
    result_str.append(f"Time: {cur_time}")
    result_str.append(f"Model Path: {model_ckpt_path}")
    result_str.append(f"Model Type: {config['model']}")
    result_str.append(f"Trained Epochs: {epoch}")
    result_str.append(f"========================================")

    # 分组打印指标，方便查看
    result_str.append("\n[Recall / Hit-Rate]")
    for k, v in final_results.items():
        if 'Recall' in k or 'Hit' in k:
            result_str.append(f"{k}: {v:.5f}")

    result_str.append("\n[NDCG / Ranking]")
    for k, v in final_results.items():
        if 'NDCG' in k or 'MRR' in k:
            result_str.append(f"{k}: {v:.5f}")

    result_str.append("\n[Tail / Long-Tail Ability] (Target is NOT in Top 1%)")
    for k, v in final_results.items():
        if 'Tail' in k:
            result_str.append(f"{k}: {v:.5f}")

    result_str.append("\n[Diversity & Novelty & Coverage]")
    for k, v in final_results.items():
        if 'Diversity' in k or 'Novelty' in k or 'Coverage' in k:
            result_str.append(f"{k}: {v:.5f}")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_str))

    # 打印到控制台
    print('\n'.join(result_str))
    print(f"\nResult saved to {output_file}")


if __name__ == "__main__":
    EXP_PATH = 'EXP_20251209_182837'
    DATA_PATH = '../../data/sbr_data_1208.pkl'
    MODEL_PATH = f'../../experiments/{EXP_PATH}/checkpoints/best_model.pth'
    OUTPUT_FILE = f'../../experiments/{EXP_PATH}/final_offline_metrics.txt'
    run_evaluation(DATA_PATH, MODEL_PATH, OUTPUT_FILE)