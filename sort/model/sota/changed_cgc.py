import torch
import torch.nn as nn

from layer.din import MultiHeadTargetAttention
from layer.dcn import CrossNetV2


class ChangedCGC(nn.Module):
    def __init__(self, feature_dict, max_seq_len=30,
                 num_specific_experts=2, num_shared_experts=2, expert_hidden_dim=256,
                 task_hidden_dims=[128, 64], num_tasks=2, drop_rate=0.3, device='cpu',
                 use_dcn=True, use_ppnet=True):  # <-- 新增开关参数
        super(ChangedCGC, self).__init__()

        self.feature_dict = feature_dict
        self.num_tasks = num_tasks
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts

        self.use_dcn = use_dcn
        self.use_ppnet = use_ppnet

        # --- Step 1: 基础表示层 (Embedding Layer) ---
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, feat_dim)
            for col, (vocab_size, feat_dim, _) in feature_dict.items()
        })

        self.item_emb_dim = feature_dict['item_id'][1]
        self.bhv_emb = nn.Embedding(5, self.item_emb_dim)
        self.pos_emb = nn.Embedding(max_seq_len, self.item_emb_dim)

        self.item_idx = feature_dict['item_id'][2]
        self.user_idx = feature_dict['user_id'][2]
        self.user_emb_dim = feature_dict['user_id'][1]

        # --- Step 2: 序列兴趣抽取层 (Sequence Modeling) ---
        self.target_attention = MultiHeadTargetAttention(emb_dim=self.item_emb_dim, num_heads=4)

        # --- Step 3: 特征交叉层 (纯 DCN-v2) ---
        total_sparse_dim = sum(feat_dim for _, feat_dim, _ in feature_dict.values())
        concat_dim = total_sparse_dim + self.item_emb_dim

        # 根据是否使用 DCN 动态计算输入到 Expert 的维度
        if self.use_dcn:
            self.cross_net = CrossNetV2(in_features=concat_dim, layer_num=3)
            shared_dim = concat_dim * 2  # concat(x_in, x_cross)
        else:
            shared_dim = concat_dim  # 直接使用 x_in

        # --- Step 4: 多任务路由层 (CGC - 单层 PLE) ---
        self.shared_experts = nn.ModuleList([
            self._build_expert(shared_dim, expert_hidden_dim, drop_rate)
            for _ in range(self.num_shared_experts)
        ])

        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                self._build_expert(shared_dim, expert_hidden_dim, drop_rate)
                for _ in range(self.num_specific_experts)
            ]) for _ in range(self.num_tasks)
        ])

        num_experts_per_task = self.num_specific_experts + self.num_shared_experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, num_experts_per_task),
                nn.Softmax(dim=-1)
            ) for _ in range(self.num_tasks)
        ])

        # --- Step 5: 个性化任务塔 (PPNet-style Task Towers) ---
        self.task_towers = nn.ModuleList()
        if self.use_ppnet:
            self.ppnet_gates = nn.ModuleList()
        self.task_out_layers = nn.ModuleList()

        for _ in range(num_tasks):
            tower = []
            input_dim = expert_hidden_dim
            for out_dim in task_hidden_dims:
                tower.append(nn.Linear(input_dim, out_dim))
                tower.append(nn.BatchNorm1d(out_dim))
                tower.append(nn.ReLU())
                tower.append(nn.Dropout(drop_rate))
                input_dim = out_dim
            self.task_towers.append(nn.Sequential(*tower))

            if self.use_ppnet:
                self.ppnet_gates.append(nn.Linear(self.user_emb_dim, task_hidden_dims[-1]))

            self.task_out_layers.append(nn.Linear(task_hidden_dims[-1], 1))

        if self.use_ppnet:
            for gate in self.ppnet_gates:
                nn.init.zeros_(gate.weight)
                if gate.bias is not None:
                    nn.init.zeros_(gate.bias)

    def _build_expert(self, in_dim, out_dim, drop_rate):
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, out_dim),
            nn.ReLU()
        )

    def forward(self, sparse_x, seq_item, seq_bhv):
        B = sparse_x.size(0)

        # --- Step 1: 基础表示 ---
        sparse_embs = []
        for col, (_, _, idx) in self.feature_dict.items():
            sparse_embs.append(self.embeddings[col](sparse_x[:, idx]))

        user_emb = self.embeddings['user_id'](sparse_x[:, self.user_idx])
        target_item_emb = self.embeddings['item_id'](sparse_x[:, self.item_idx]).unsqueeze(1)

        # --- Step 2: 序列兴趣抽取 ---
        seq_item_e = self.embeddings['item_id'](seq_item)
        seq_bhv_e = self.bhv_emb(seq_bhv)

        seq_len = seq_item.size(1)
        positions = torch.arange(seq_len, device=seq_item.device).unsqueeze(0).expand(B, -1)
        seq_pos_e = self.pos_emb(positions)

        seq_e_fused = seq_item_e + seq_bhv_e + seq_pos_e
        mask = (seq_item > 0)
        hist_emb = self.target_attention(target_item_emb, seq_e_fused, mask)

        # --- Step 3: 特征交叉 ---
        flat_sparse = torch.cat(sparse_embs, dim=-1)
        x_in = torch.cat([flat_sparse, hist_emb], dim=-1)

        # 动态处理 DCN 输出
        if self.use_dcn:
            x_cross = self.cross_net(x_in)
            x_shared = torch.cat([x_in, x_cross], dim=-1)
        else:
            x_shared = x_in

        # --- Step 4: CGC (单层 PLE) ---
        shared_expert_outputs = [expert(x_shared) for expert in self.shared_experts]

        task_inputs = []
        for i in range(self.num_tasks):
            specific_expert_outputs = [expert(x_shared) for expert in self.specific_experts[i]]
            all_experts_for_task = torch.stack(specific_expert_outputs + shared_expert_outputs, dim=1)
            gate_weights = self.gates[i](x_shared).unsqueeze(-1)
            task_in = torch.sum(all_experts_for_task * gate_weights, dim=1)
            task_inputs.append(task_in)

        # --- Step 5: PPNet 个性化任务塔 ---
        task_outputs = []
        for i in range(self.num_tasks):
            tower_out = self.task_towers[i](task_inputs[i])

            # 动态处理 PPNet 门控
            if self.use_ppnet:
                user_gate = 2.0 * torch.sigmoid(self.ppnet_gates[i](user_emb))
                tower_out_personalized = tower_out * user_gate
            else:
                tower_out_personalized = tower_out

            logit = self.task_out_layers[i](tower_out_personalized)
            task_outputs.append(logit.squeeze(-1))

        return torch.stack(task_outputs, dim=1)