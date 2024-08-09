# @Time   : 2022/3/8
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com
import copy
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv, ResidualVectorQuantizer, ProductVectorQuantizer


class CoGCL(GeneralGraphRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CoGCL, self).__init__(config, dataset)


        self.dataset = dataset
        if config['sim_cl_weight'] > 0:
            self.dataset.get_same_targe_users_items()
        setattr(self.dataset, 'code_similar_users', [[]] * self.n_users)
        setattr(self.dataset, 'code_similar_items', [[]] * self.n_items)

        self.config = config

        self.AUG_USER_ID  = "aug_" + self.USER_ID
        self.AUG_ITEM_ID = "aug_" + self.ITEM_ID
        
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        self.user_code_num = config["user_code_num"]
        self.item_code_num = config["item_code_num"]
        self.user_code_size = config["user_code_size"]
        self.item_code_size = config["item_code_size"]
        self.n_user_codes = self.user_code_num * self.user_code_size
        self.n_item_codes = self.item_code_num * self.item_code_size

        self.code_dist = config["code_dist"]
        self.code_dist_tau = config["code_dist_tau"]
        self.code_batch_size = config["code_batch_size"]
        self.vq_loss_weight = config['vq_loss_weight']
        self.vq_type = config['vq_type']
        self.vq_ema = config['vq_ema']


        # load parameters info
        self.embedding_size = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of LightGCN
        
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        
        self.cl_weight = config['cl_weight']
        self.sim_cl_weight = config['sim_cl_weight']
        self.cl_tau = config['cl_tau']
        
        self.graph_replace_p = config['graph_replace_p']
        self.graph_add_p = config['graph_add_p']

        self.drop_p = config['drop_p']
        self.drop_fwd = config['drop_fwd']

        self.data_aug_delay = config['data_aug_delay']
        self.epoch_num = 0


        if self.vq_type.lower() == 'rq':
            self.user_vq = ResidualVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size, 
                                                   codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ResidualVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                    codebook_dim=self.embedding_size, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        elif self.vq_type.lower() == 'pq':
            assert self.embedding_size % self.user_code_num == 0
            assert self.embedding_size % self.item_code_num == 0
            self.user_vq = ProductVectorQuantizer(codebook_num=self.user_code_num, codebook_size=self.user_code_size,
                                                  codebook_dim=self.embedding_size//self.user_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
            self.item_vq = ProductVectorQuantizer(codebook_num=self.item_code_num, codebook_size=self.item_code_size,
                                                  codebook_dim=self.embedding_size//self.item_code_num, dist=self.code_dist, tau=self.code_dist_tau, vq_ema=self.vq_ema)
        else:
            raise NotImplementedError



        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_size
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.embedding_size
        )

        self.user_code_embedding = torch.nn.Embedding(
            num_embeddings=self.n_user_codes, embedding_dim=self.embedding_size
        )
        self.item_code_embedding = torch.nn.Embedding(
            num_embeddings=self.n_item_codes, embedding_dim=self.embedding_size
        )


        self.gcn_conv = LightGCNConv(dim=self.embedding_size)

        self.dropout = torch.nn.Dropout(self.drop_p)


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.aug_edge_index_1 = None
        self.aug_edge_weight_1 = None
        self.aug_edge_index_2 = None
        self.aug_edge_weight_2 = None

        self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight = None, None
        self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight = self.inter_graph_aug()


        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']



    def train(self, mode: bool = True):
        r"""Override train method of base class. The subgraph is reconstructed each time it is called.

        """
        
        if mode and self.epoch_num >= self.data_aug_delay:
            if self.epoch_num == self.data_aug_delay:
                print("Start Data Augmentation")
            self.graph_augment()

        if mode:
            self.epoch_num += 1

        T = super().train(mode=mode)

        return T


    def graph_augment(self):

        all_user_codes, all_item_codes = self.get_all_codes()

        # 0 for replace, 1 for add
        aug_types = random.choices([0, 1], k=2)
        if self.sim_cl_weight > 0:
            similar_users, similar_items = self.get_share_codes_info(all_user_codes, all_item_codes)
            setattr(self.dataset, 'code_similar_users', similar_users)
            setattr(self.dataset, 'code_similar_items', similar_items)

        all_user_codes = all_user_codes.detach().cpu().numpy()
        all_item_codes = all_item_codes.detach().cpu().numpy()

        self.aug_edge_index_1, self.aug_edge_weight_1 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[0])
        self.aug_edge_index_2, self.aug_edge_weight_2 = self.inter_graph_aug(all_user_codes, all_item_codes, aug_types[1])


    @torch.no_grad()
    def get_all_codes(self):

        self.user_vq.eval()
        self.item_vq.eval()

        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user_all_embeddings = self.restore_user_e
        item_all_embeddings = self.restore_item_e

        start, batch_size = 0, self.code_batch_size
        all_user_codes = []
        while start < self.n_users:
            batch_user_embs = user_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_user_embs, self.user_vq)
            all_user_codes.append(codes)
            start += batch_size
        all_user_codes = torch.cat(all_user_codes, dim=0)
        user_offset = torch.arange(self.user_code_num, dtype=torch.long, device=self.device) * self.user_code_size
        all_user_codes = all_user_codes + user_offset.unsqueeze(0)

        start, batch_size = 0, self.code_batch_size
        all_item_codes = []
        while start < self.n_items:
            batch_item_embs = item_all_embeddings[start: start + batch_size]
            _, codes = self.emb_quantize(batch_item_embs, self.item_vq)
            all_item_codes.append(codes)
            start += batch_size
        all_item_codes = torch.cat(all_item_codes, dim=0)
        item_offset = torch.arange(self.item_code_num, dtype=torch.long, device=self.device) * self.item_code_size
        all_item_codes = all_item_codes + item_offset.unsqueeze(0)

        return all_user_codes, all_item_codes

    def get_share_codes_info(self, all_user_codes, all_item_codes):

        start, batch_size = 0, self.code_batch_size
        similar_users = [[]] * self.n_users
        while start < self.n_users:
            batch_ucodes = all_user_codes[start: start + batch_size]
            batch_sim = []
            k = 0
            while k < self.n_users:
                block = all_user_codes[k: k + batch_size]
                # B, 4, 1 == 1, 4, B   ->   B, 4, B   ->  B, B
                sim = (batch_ucodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            # # B, 4, 1 == 1, 4, N   ->   B, 4, N   ->  B, N
            # sim = (batch_ucodes.unsqueeze(-1) == all_user_codes.T.unsqueeze(0)).to(torch.int).sum(dim=1)
            batch_sim = torch.cat(batch_sim, dim=1)
            batch_sim[:, 0] = 0

            for i, sim in enumerate(batch_sim):
                uid = start + i
                sim_index = torch.where(sim >= self.user_code_num - 1)[0]
                similar_users[uid] = sim_index.cpu().numpy()

            start += batch_size



        start, batch_size = 0, self.code_batch_size
        similar_items = [[]] * self.n_items
        while start < self.n_items:
            batch_icodes = all_item_codes[start: start + batch_size]
            batch_sim = []
            k = 0
            while k < self.n_items:
                block = all_item_codes[k: k + batch_size]
                sim = (batch_icodes.unsqueeze(-1) == block.T.unsqueeze(0)).to(torch.int).sum(dim=1)
                batch_sim.append(sim)
                k += batch_size
            batch_sim = torch.cat(batch_sim, dim=1)
            batch_sim[:, 0] = 0
            for i, sim in enumerate(batch_sim):
                iid = start + i
                sim_index = torch.where(sim >= self.item_code_num - 1)[0]
                similar_items[iid] = sim_index.cpu().numpy()

            start += batch_size

        return similar_users, similar_items

    def inter_graph_aug(self, all_user_codes=None, all_item_codes=None, aug_type=2):



        if all_user_codes is None or all_item_codes is None:
            assert aug_type==2

        if aug_type == 2 and self.non_graph_aug_edge_index is not None and self.non_graph_aug_edge_weight is not None:
            return self.non_graph_aug_edge_index, self.non_graph_aug_edge_weight

        row = self._user
        col = self._item

        all_idx = np.arange(len(row))
        if aug_type == 0: # replace
            aug_num = int(len(all_idx) * self.graph_replace_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
            keep_idx = list( set(all_idx) - set(user_aug_idx) - set(item_aug_idx) )

        elif aug_type==1: # add
            keep_idx = all_idx
            aug_num = int(len(all_idx) * self.graph_add_p)
            user_aug_idx = np.random.choice(len(row), aug_num, replace=False)
            item_aug_idx = np.random.choice(len(col), aug_num, replace=False)
        else: # non graph aug
            keep_idx = all_idx
            user_aug_idx = []
            item_aug_idx = []

        keep_row = row[keep_idx]
        keep_col = col[keep_idx] + self.n_users + self.n_user_codes


        if aug_type == 2:
            aug_inter_row = []
            aug_inter_col = []
        else:
            aug_inter_row = []
            aug_inter_col = []

            aug_row = row[item_aug_idx]
            aug_col = col[item_aug_idx]
            for user, item in zip(aug_row, aug_col):
                item_codes = all_item_codes[item] + self.n_items + self.n_users + self.n_user_codes
                item_codes = item_codes.tolist()
                aug_inter_row.extend([user]* len(item_codes))
                aug_inter_col.extend(item_codes)

            aug_row = row[user_aug_idx]
            aug_col = col[user_aug_idx]
            for user, item in zip(aug_row, aug_col):
                user_codes = all_user_codes[user] + self.n_users
                item = item + self.n_users + self.n_user_codes
                user_codes = user_codes.tolist()
                aug_inter_row.extend(user_codes)
                aug_inter_col.extend([item] * len(user_codes))


        all_row = torch.tensor(keep_row.numpy().tolist() + aug_inter_row).to(keep_row.dtype)
        all_col = torch.tensor(keep_col.numpy().tolist() + aug_inter_col).to(keep_col.dtype)

        edge_index1 = torch.stack([all_row, all_col])
        edge_index2 = torch.stack([all_col, all_row])
        aug_edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        aug_edge_weight = torch.ones(aug_edge_index.size(1))

        num_nodes = self.n_users + self.n_user_codes + self.n_items + self.n_item_codes

        aug_edge_index, aug_edge_weight = gcn_norm(aug_edge_index.to(self.device),
                                                   aug_edge_weight.to(self.device),
                                                   num_nodes, add_self_loops=False)


        return aug_edge_index, aug_edge_weight

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_w_codes_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        user_code_embeddings = self.user_code_embedding.weight
        item_code_embeddings = self.item_code_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, user_code_embeddings, item_embeddings, item_code_embeddings], dim=0)
        return ego_embeddings

    def forward(self, drop=False):
        embeddings = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            if drop:
                embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings
    
    def aug_forward(self, aug_edge_index=None, aug_edge_weight=None):

        if aug_edge_index is None or aug_edge_weight is None:
            aug_edge_index = self.non_graph_aug_edge_index
            aug_edge_weight = self.non_graph_aug_edge_weight

        embeddings = self.get_w_codes_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            embeddings = self.dropout(embeddings)
            embeddings = self.gcn_conv(embeddings, aug_edge_index, aug_edge_weight)
            embeddings_list.append(embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users + self.n_user_codes, self.n_items + self.n_item_codes]
        )

        return user_all_embeddings, item_all_embeddings


    def quantize(self, users, items, user_embeddings=None, item_embeddings=None):

        if user_embeddings is None or item_embeddings is None:
            user_embs = self.user_embedding(users)
            item_embs = self.item_embedding(items)
        else:
            user_embs = user_embeddings[users]
            item_embs = item_embeddings[items]
        
        user_vq_loss, _ = self.emb_quantize(user_embs, self.user_vq)
        item_vq_loss, _ = self.emb_quantize(item_embs, self.item_vq)
        
        return user_vq_loss + item_vq_loss
    
    
    def emb_quantize(self, x, vq_layer):

        x = x.detach()

        x_q, mean_com_loss, all_codes = vq_layer(x)

        if self.code_dist.lower() == 'l2':
            recon_loss = F.mse_loss(x_q, x)
        elif self.code_dist.lower() == 'cos':
            recon_scores = torch.matmul(F.normalize(x_q, dim=-1), F.normalize(x, dim=-1).t()) / self.code_dist_tau
            recon_labels = torch.arange(x.size(0), dtype=torch.long).to(x.device)
            recon_loss = F.cross_entropy(recon_scores, recon_labels)
        else:
            raise NotImplementedError

        loss = recon_loss + mean_com_loss

        return loss, all_codes


    def calculate_cl_loss(self, x1, x2):
        assert x1.size(0) == x2.size(0)
        B = x1.size(0)
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)

        sim_12 = torch.mm(x1, x2.T)
        sim_21 = sim_12.T
        sim_11 = torch.mm(x1, x1.T)
        sim_22 = torch.mm(x2, x2.T)

        sim_12 = torch.exp(sim_12 / self.cl_tau)
        sim_21 = torch.exp(sim_21 / self.cl_tau)
        sim_11 = torch.exp(sim_11 / self.cl_tau)
        sim_22 = torch.exp(sim_22 / self.cl_tau)


        loss_12 = -torch.log(
            sim_12.diag()
            / (sim_12.sum(1) + sim_11.sum(1) - sim_11.diag())
        )
        loss_21 = -torch.log(
            sim_21.diag()
            / (sim_21.sum(1) + sim_22.sum(1) - sim_22.diag())
        )


        loss = 0.5 * (loss_12 + loss_21).mean()

        return loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(drop=self.drop_fwd)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_id_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_id_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_id_scores, neg_id_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user).view(-1, self.embedding_size)
        pos_ego_embeddings = self.item_embedding(pos_item).view(-1, self.embedding_size)
        neg_ego_embeddings = self.item_embedding(neg_item).view(-1, self.embedding_size)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        user = torch.unique(interaction[self.USER_ID], sorted=True)
        pos_item = torch.unique(interaction[self.ITEM_ID], sorted=True)

        vq_loss = self.quantize(user, pos_item, user_all_embeddings, item_all_embeddings)

        user_aug1_embeddings, item_aug1_embeddings = self.aug_forward(self.aug_edge_index_1,self.aug_edge_weight_1)
        user_aug2_embeddings, item_aug2_embeddings = self.aug_forward(self.aug_edge_index_2,self.aug_edge_weight_2)
        u_x1, u_x2 = user_aug1_embeddings[user], user_aug2_embeddings[user]
        i_x1, i_x2 = item_aug1_embeddings[pos_item], item_aug2_embeddings[pos_item]

        cl_loss = self.calculate_cl_loss(u_x1, u_x2) + self.calculate_cl_loss(i_x1, i_x2)

        if self.sim_cl_weight > 0:
            aug_users = interaction[self.AUG_USER_ID]
            aug_items = interaction[self.AUG_ITEM_ID]

            u_x3 = user_all_embeddings[aug_users]
            i_x3 = item_all_embeddings[aug_items]

            sim_cl_loss_1 = self.calculate_cl_loss(u_x1, u_x3) + self.calculate_cl_loss(i_x1, i_x3)
            sim_cl_loss_2 = self.calculate_cl_loss(u_x2, u_x3) + self.calculate_cl_loss(i_x2, i_x3)

            sim_cl_loss = 0.5 * (sim_cl_loss_1 + sim_cl_loss_2)
        else:
            sim_cl_loss = torch.zeros(1).to(self.device)


        return (mf_loss, self.reg_weight * reg_loss, self.vq_loss_weight * vq_loss,
                self.cl_weight * cl_loss, self.sim_cl_weight * sim_cl_loss)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

