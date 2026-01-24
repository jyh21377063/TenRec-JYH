import torch
import torch.nn as nn

from layer.din import DINAttentionLayer
from layer.dcn import CrossNetV2


class ESMM(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, hidden_dim=[128, 128], dropouts=[0.5, 0.5],
                 output_size=1, num_task=2):
        super(ESMM, self).__init__()

        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(user_feature_dict) - user_cate_feature_nums) + (
                                  len(item_feature_dict) - item_cate_feature_nums)

        # esmm 独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        hidden = torch.cat([user_embed, item_embed], axis=1).float()

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = hidden
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
        # task_outputs[1] = task_outputs[0] * task_outputs[1]
        return task_outputs


class ESMM_SEQ(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, hidden_dim=[128, 128], dropouts=[0.5, 0.5],
                 output_size=1, num_task=2):
        super(ESMM_SEQ, self).__init__()

        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(user_feature_dict) - user_cate_feature_nums) + \
                    (len(item_feature_dict) - item_cate_feature_nums)

        # 加上序列 Avg Pooling 后的维度
        hidden_size = input_dim + emb_dim

        # esmm 独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        item_id_emb_layer = getattr(self, 'item_id')  # 复用 item_id embedding

        seq_emb = item_id_emb_layer(x_seq.long())
        mask = (x_seq > 0).float().unsqueeze(-1)
        seq_sum = torch.sum(seq_emb * mask, dim=1)
        seq_avg = seq_sum / (torch.sum(mask, dim=1) + 1e-8)

        hidden = torch.cat([user_embed, item_embed, seq_avg], axis=1).float()

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = hidden
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
        # task_outputs[1] = task_outputs[0] * task_outputs[1]
        return task_outputs


class ESMM_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, hidden_dim=[128, 128], dropouts=[0.5, 0.5], din_hidden_dim=[256, 128],
                 output_size=1, num_task=2):
        super(ESMM_DIN, self).__init__()

        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(user_feature_dict) - user_cate_feature_nums) + \
                    (len(item_feature_dict) - item_cate_feature_nums)

        # 加上序列 Avg Pooling 后的维度
        hidden_size = input_dim + emb_dim
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        # esmm 独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [hidden_size] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        item_id_emb_layer = getattr(self, 'item_id')  # 复用 item_id embedding

        seq_emb = item_id_emb_layer(x_seq.long())
        item_id_idx = self.item_feature_dict['item_id'][1]
        target_item_emb = item_id_emb_layer(x[:, item_id_idx].long())  # (Batch, Emb_Dim)

        # 3. 准备 Mask
        mask = (x_seq > 0).float()  # (Batch, Seq_Len)
        seq = self.din_attention(target_item_emb, seq_emb, mask)

        hidden = torch.cat([user_embed, item_embed, seq], axis=1).float()

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = hidden
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)
        # task_outputs[1] = task_outputs[0] * task_outputs[1]
        return task_outputs

class ESMM_DCN_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128,
                 hidden_dim=[128, 64], dropouts=[0.5, 0.5], din_hidden_dim=[256, 128],
                 dcn_layers=3, output_size=1, num_task=2):
        super(ESMM_DCN_DIN, self).__init__()

        # === 1. 参数校验与保存 ===
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("Feature dicts must not be None")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        self.embeddings = nn.ModuleDict()
        self.dense_trans = nn.ModuleDict()

        all_feats = {**user_feature_dict, **item_feature_dict}
        self.num_sparse = 0
        self.num_dense = 0

        for name, num in all_feats.items():
            if num[0] > 1:
                self.embeddings[name] = nn.Embedding(num[0], emb_dim)
                self.num_sparse += 1
            else:
                self.dense_trans[name] = nn.Linear(1, emb_dim)
                self.num_dense += 1

        # === 3. 核心组件初始化 ===

        # A. DIN 组件
        # 必须确保 item_id 在字典中
        if 'item_id' not in self.item_feature_dict:
            raise ValueError("DIN requires 'item_id' feature")
        self.target_item_col = "item_id"
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        # B. DCN 组件
        # DCN 输入维度 = 所有特征 flattened 的长度
        total_static_dim = (self.num_sparse + self.num_dense) * emb_dim
        self.dcn_input_dim = total_static_dim + emb_dim
        self.dcn = CrossNetV2(input_dim=self.dcn_input_dim, num_layers=dcn_layers)

        # === 4. 计算 Task Tower 的输入维度 ===
        # 输入 = [原始特征 Flatten] + [DIN 序列兴趣] + [DCN 交叉特征]
        # 注意：DCN 输出维度等于输入维度
        tower_input_dim = self.dcn_input_dim + self.dcn_input_dim

        # === 5. ESMM Task Towers ===
        self.ctr_tower = self._make_tower(tower_input_dim, hidden_dim, dropouts, output_size)
        self.cvr_tower = self._make_tower(tower_input_dim, hidden_dim, dropouts, output_size)

    def _make_tower(self, input_dim, hidden_dim, dropouts, output_size):
        layers = []
        for h, drop in zip(hidden_dim, dropouts):
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            input_dim = h
        layers.append(nn.Linear(input_dim, output_size))
        return nn.Sequential(*layers)

    def forward(self, x, x_seq):
        # x: [Batch, Num_Features] (全是离散特征的 Index)

        # === 1. 获取 Embeddings ===
        embed_list = []
        all_feats = {**self.user_feature_dict, **self.item_feature_dict}

        for name, num in all_feats.items():
            if num[0] > 1:
                embed_list.append(self.embeddings[name](x[:, num[1]].long()))
            else:
                val = x[:, num[1]].unsqueeze(1).float()
                embed_list.append(self.dense_trans[name](val))

        # [Batch, Total_Feats * Emb_Dim] -> 用于 DCN 和 拼接
        flat_input_emb = torch.cat(embed_list, dim=1)

        # --- B. DIN 序列特征提取 ---
        # 动态获取 item_id 的索引
        item_col_idx = self.item_feature_dict[self.target_item_col][1]
        target_item_emb = self.embeddings[self.target_item_col](x[:, item_col_idx].long())
        seq_emb = self.embeddings[self.target_item_col](x_seq.long())
        mask = (x_seq > 0).float()

        din_out = self.din_attention(target_item_emb, seq_emb, mask)

        dcn_in = torch.cat([flat_input_emb, din_out], dim=1)
        dcn_out = self.dcn(dcn_in)

        # --- D. 拼接最终特征 ---
        final_input = torch.cat([dcn_in, dcn_out], dim=1)

        # --- E. 双塔输出 ---
        ctr_logit = self.ctr_tower(final_input)
        cvr_logit = self.cvr_tower(final_input)

        # --- F. 返回 List (保持兼容) ---
        # 约定：第一个是 CTR，第二个是 CVR
        return [ctr_logit, cvr_logit]