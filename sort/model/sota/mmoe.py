import torch
import torch.nn as nn

from layer.din import MultiHeadTargetAttention
from layer.dcn import CrossNetV2


class AdvancedMMOE(nn.Module):
    def __init__(self, feature_dict, max_seq_len=30, emb_dim=64,
                 num_experts=4, mmoe_hidden_dim=256,
                 task_hidden_dims=[128, 64], num_tasks=2, drop_rate=0.3, device='cpu'):
        super(AdvancedMMOE, self).__init__()

        self.feature_dict = feature_dict
        self.num_tasks = num_tasks

        # --- Step 1: 基础表示层 (Embedding Layer) ---
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, feat_dim)
            for col, (vocab_size, feat_dim, _) in feature_dict.items()
        })

        # 序列专属 Embedding
        self.item_emb_dim = feature_dict['item_id'][1]
        # 0: padding, 1: click, 2: like, 3: follow, 4: share (总共5个)
        self.bhv_emb = nn.Embedding(5, self.item_emb_dim)
        # 位置编码
        self.pos_emb = nn.Embedding(max_seq_len, self.item_emb_dim)

        # 提取 item_id 和 user_id 在 sparse_x 中的索引，方便后续提取
        self.item_idx = feature_dict['item_id'][2]
        self.user_idx = feature_dict['user_id'][2]
        self.user_emb_dim = feature_dict['user_id'][1] # 提取用户实际的emb_dim

        # --- Step 2: 序列兴趣抽取层 (Sequence Modeling) ---
        self.target_attention = MultiHeadTargetAttention(emb_dim=self.item_emb_dim, num_heads=4)

        # --- Step 3: 特征交叉层 (Parallel DCN-v2 + DNN) ---
        total_sparse_dim = sum(feat_dim for _, feat_dim, _ in feature_dict.values())
        concat_dim = total_sparse_dim + emb_dim

        self.cross_net = CrossNetV2(in_features=concat_dim, layer_num=3)
        self.deep_net = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        shared_dim = concat_dim + 256  # Cross(不变) + Deep(256)

        # --- Step 4: 多任务路由层 (MMoE) ---
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, mmoe_hidden_dim),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])

        # --- Step 5: 个性化任务塔 (PPNet-style Task Towers) ---
        self.task_towers = nn.ModuleList()
        self.ppnet_gates = nn.ModuleList()
        self.task_out_layers = nn.ModuleList()

        for _ in range(num_tasks):
            # Task Tower
            tower = []
            input_dim = mmoe_hidden_dim
            for out_dim in task_hidden_dims:
                tower.append(nn.Linear(input_dim, out_dim))
                tower.append(nn.BatchNorm1d(out_dim))
                tower.append(nn.ReLU())
                tower.append(nn.Dropout(drop_rate))
                input_dim = out_dim
            self.task_towers.append(nn.Sequential(*tower))

            # PPNet Gate: 输入为 user_emb，输出维度与 Task Tower 最后一层隐藏层对齐
            self.ppnet_gates.append(nn.Linear(self.user_emb_dim, task_hidden_dims[-1]))

            # 最终输出层 (Binary Classification Logits)
            self.task_out_layers.append(nn.Linear(task_hidden_dims[-1], 1))

        for gate in self.ppnet_gates:
            nn.init.zeros_(gate.weight)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def forward(self, sparse_x, seq_item, seq_bhv):
        B = sparse_x.size(0)

        # --- Step 1: 基础表示 ---
        sparse_embs = []
        # 前向传播时，解包三元组获取索引
        for col, (_, _, idx) in self.feature_dict.items():
            sparse_embs.append(self.embeddings[col](sparse_x[:, idx]))

        user_emb = self.embeddings['user_id'](sparse_x[:, self.user_idx])
        target_item_emb = self.embeddings['item_id'](sparse_x[:, self.item_idx]).unsqueeze(1)  # [B, 1, E]

        # --- Step 2: 序列兴趣抽取 ---
        seq_item_e = self.embeddings['item_id'](seq_item)  # [B, L, E]
        seq_bhv_e = self.bhv_emb(seq_bhv)  # [B, L, E]

        # 动态生成位置索引并 lookup
        seq_len = seq_item.size(1)
        positions = torch.arange(seq_len, device=seq_item.device).unsqueeze(0).expand(B, -1)
        seq_pos_e = self.pos_emb(positions)  # [B, L, E]

        # 融合: E_seq = Embedding(seq_item) + Embedding(seq_bhv) + Embedding(pos)
        seq_e_fused = seq_item_e + seq_bhv_e + seq_pos_e

        # 计算 Mask
        mask = (seq_item > 0)  # [B, L] Bool Tensor

        # DIN Attention 聚合
        hist_emb = self.target_attention(target_item_emb, seq_e_fused, mask)

        # --- Step 3: 特征交叉 ---
        # 拼接所有 Sparse Emb 和 动态历史 Emb
        flat_sparse = torch.cat(sparse_embs, dim=-1)  # [B, num_sparse * E]
        x_in = torch.cat([flat_sparse, hist_emb], dim=-1)  # [B, concat_dim]

        x_cross = self.cross_net(x_in)  # [B, concat_dim]
        x_deep = self.deep_net(x_in)  # [B, 256]

        x_shared = torch.cat([x_cross, x_deep], dim=-1)  # [B, shared_dim]

        # --- Step 4: MMoE ---
        expert_outputs = torch.stack([expert(x_shared) for expert in self.experts], dim=1)  # [B, num_experts, mmoe_dim]

        task_inputs = []
        for i in range(self.num_tasks):
            gate_weights = self.gates[i](x_shared).unsqueeze(-1)  # [B, num_experts, 1]
            task_in = torch.sum(expert_outputs * gate_weights, dim=1)  # [B, mmoe_dim]
            task_inputs.append(task_in)

        # --- Step 5: PPNet 个性化任务塔 ---
        task_outputs = []
        for i in range(self.num_tasks):
            # 过普通的 DNN Tower
            tower_out = self.task_towers[i](task_inputs[i])  # [B, task_hidden_dims[-1]]

            # 计算 User 个性化 Gate: Gate = 2 * Sigmoid(Linear(E_user))
            # 乘以 2 是为了让初始门控期望在 1 附近，便于梯度传播 (LHUC 标准做法)
            user_gate = 2.0 * torch.sigmoid(self.ppnet_gates[i](user_emb))

            # 门控融合
            tower_out_personalized = tower_out * user_gate

            # 输出 Logits
            logit = self.task_out_layers[i](tower_out_personalized)
            task_outputs.append(logit.squeeze(-1))  # [B]

        return torch.stack(task_outputs, dim=1)  # 输出维度变为 [B, num_tasks]