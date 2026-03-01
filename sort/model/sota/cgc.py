import torch
import torch.nn as nn

from layer.din import MultiHeadTargetAttention
from layer.dcn import CrossNetV2


class AdvancedCGC(nn.Module):
    def __init__(self, feature_dict, max_seq_len=30,
                 num_specific_experts=2, num_shared_experts=2, expert_hidden_dim=256,
                 task_hidden_dims=[128, 64], num_tasks=2, drop_rate=0.3, device='cpu'):
        super(AdvancedCGC, self).__init__()

        self.feature_dict = feature_dict
        self.num_tasks = num_tasks
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts

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
        self.cross_net = CrossNetV2(in_features=concat_dim, layer_num=3)

        shared_dim = concat_dim * 2

        # --- Step 4: 多任务路由层 (CGC - 单层 PLE) ---
        # 1. 构造共享专家 (Shared Experts)
        self.shared_experts = nn.ModuleList([
            self._build_expert(shared_dim, expert_hidden_dim, drop_rate)
            for _ in range(self.num_shared_experts)
        ])

        # 2. 构造任务专属专家 (Task-Specific Experts)
        # 结构为 ModuleList(ModuleList(Expert)), 维度为 [num_tasks, num_specific_experts]
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                self._build_expert(shared_dim, expert_hidden_dim, drop_rate)
                for _ in range(self.num_specific_experts)
            ]) for _ in range(self.num_tasks)
        ])

        # 3. 构造任务门控 (Task Gates)
        # CGC 中，每个任务的门控掌管: 自己的 specific 专家 + 所有的 shared 专家
        num_experts_per_task = self.num_specific_experts + self.num_shared_experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, num_experts_per_task),
                nn.Softmax(dim=-1)
            ) for _ in range(self.num_tasks)
        ])

        # --- Step 5: 个性化任务塔 (PPNet-style Task Towers) ---
        self.task_towers = nn.ModuleList()
        self.ppnet_gates = nn.ModuleList()
        self.task_out_layers = nn.ModuleList()

        for _ in range(num_tasks):
            # Task Tower
            tower = []
            input_dim = expert_hidden_dim
            for out_dim in task_hidden_dims:
                tower.append(nn.Linear(input_dim, out_dim))
                tower.append(nn.BatchNorm1d(out_dim))
                tower.append(nn.ReLU())
                tower.append(nn.Dropout(drop_rate))
                input_dim = out_dim
            self.task_towers.append(nn.Sequential(*tower))

            # PPNet Gate
            self.ppnet_gates.append(nn.Linear(self.user_emb_dim, task_hidden_dims[-1]))
            # 最终输出层
            self.task_out_layers.append(nn.Linear(task_hidden_dims[-1], 1))

        for gate in self.ppnet_gates:
            nn.init.zeros_(gate.weight)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def _build_expert(self, in_dim, out_dim, drop_rate):
        """辅助函数：构建单个 Expert 网络"""
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

        x_cross = self.cross_net(x_in)
        x_shared = torch.cat([x_in, x_cross], dim=-1)

        # --- Step 4: CGC (单层 PLE) ---
        # 1. 计算所有 Shared Experts 的输出
        # shared_expert_outputs: List of Tensors, length = num_shared_experts
        shared_expert_outputs = [expert(x_shared) for expert in self.shared_experts]

        task_inputs = []
        for i in range(self.num_tasks):
            # 2. 计算当前任务专属 Specific Experts 的输出
            specific_expert_outputs = [expert(x_shared) for expert in self.specific_experts[i]]

            # 3. 将当前任务的专属专家和共享专家拼接起来
            # [B, num_specific_experts + num_shared_experts, expert_dim]
            all_experts_for_task = torch.stack(specific_expert_outputs + shared_expert_outputs, dim=1)

            # 4. 计算当前任务的 Gate 权重
            gate_weights = self.gates[i](x_shared).unsqueeze(-1)  # [B, num_experts_per_task, 1]

            # 5. 加权求和得到当前任务的输入
            task_in = torch.sum(all_experts_for_task * gate_weights, dim=1)  # [B, expert_dim]
            task_inputs.append(task_in)

        # --- Step 5: PPNet 个性化任务塔 ---
        task_outputs = []
        for i in range(self.num_tasks):
            tower_out = self.task_towers[i](task_inputs[i])
            user_gate = 2.0 * torch.sigmoid(self.ppnet_gates[i](user_emb))
            tower_out_personalized = tower_out * user_gate

            logit = self.task_out_layers[i](tower_out_personalized)
            task_outputs.append(logit.squeeze(-1))

        return torch.stack(task_outputs, dim=1)