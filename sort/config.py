import torch


class Config:
    __name = "MMOE"
    experiment_name = f"{__name}_sota"
    model_name = __name
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_model = False

    raw_data_path = '../../data/ctr_data_1M_mapped.csv'
    data_dir = '../../data/ctr_data_1224'
    log_dir = '../../experiments'

    sparse_features = ["user_id", "item_id", "video_category", "gender", "age",
    ]
    target_cols = ['click', 'like', 'follow', 'share']

    item_feature_names = ['item_id', 'video_category']
    user_feature_names = ['user_id', 'gender', 'age']
    use_targets = ['click', 'like']

    data_num_workers = 8


    # --- 训练参数 ---
    # 'manual' 表示手动静态加权, 'auto' 表示自动不确定性加权
    loss_weight_mode = 'manual'
    # 仅在 manual 模式下生效，需与 use_targets 列表对应 (比如 CTR 和 CVR)
    task_weights = [1.0,3.5]
    epochs = 30
    batch_size = 10240
    lr = 1e-4
    weight_decay = 1e-5
    emb_dim = 64
    max_seq_len = 30
    patience = 5