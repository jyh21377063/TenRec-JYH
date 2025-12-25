# config.py
import torch


class Config:
    # 可选: "MMOE", "MMOE_SEQ", "PLE_SEQ", "ESMM", "ESMM_SEQ"
    __name = "MMOE"
    experiment_name = f"{__name}_v1"
    model_name = __name
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_data_path = '../../data/ctr_data_1M_mapped.csv'
    data_dir = '../../data/ctr_data_1224'
    log_dir = '../../experiments'

    sparse_features = ["user_id", "item_id", "video_category", "gender", "age",
                       # "hist_1", "hist_2", "hist_3", "hist_4", "hist_5",
                       # "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"
    ]
    target_cols = ['click', 'like', 'follow', 'share']

    item_feature_names = ['item_id', 'video_category']
    user_feature_names = ['user_id', 'gender', 'age']
    use_targets = ['click', 'like']

    # 目前就仅仅支持用5090 90G了！
    data_num_workers = 8


    # --- 模型参数 ---
    emb_dim = 64
    n_expert = 3
    mmoe_hidden_dim = 128
    hidden_dim = [128, 64]
    dropouts = [0.2, 0.2]
    # PLE专属
    n_specific_experts = 2  # 每个任务私有的专家数
    n_shared_experts = 1  # 所有任务共享的专家数
    # din的隐藏层
    din_hidden_dim = [256, 128]

    # --- 训练参数 ---
    epochs = 20
    batch_size = 10240
    lr = 1e-3
    weight_decay = 1e-5
    patience = 3