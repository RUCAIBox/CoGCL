import json
import os
import random
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
try:
    from torch_sparse import SparseTensor
    is_sparse = True
except ImportError:
    is_sparse = False

from recbole.data.dataset import SequentialDataset
from recbole.data.dataset import Dataset as RecBoleDataset
from recbole.utils import set_color, FeatureSource

import recbole
import pickle
from recbole.utils import ensure_dir


class GeneralGraphDataset(RecBoleDataset):
    def __init__(self, config):
        super().__init__(config)

        self.same_target_users = None
        self.same_target_items = None



    def get_same_targe_users_items(self):

        if self.same_target_users is not None and self.same_target_items is not None:
            return self.same_target_users, self.same_target_items

        print("Constructing same target users and same target items")
        same_target_users = []
        same_target_items = []
        users = self.inter_feat[self.uid_field].numpy()
        items = self.inter_feat[self.iid_field].numpy()

        for target_item in range(self.item_num):
            if target_item == 0:
                same_target_users.append([])
                continue
            # all users who interact with the item
            all_index = np.where(items == target_item)[0]
            all_users = users[all_index]
            all_users = np.unique(all_users)
            same_target_users.append(all_users)


        for target_user in range(self.user_num):
            if target_user == 0:
                same_target_items.append([])
                continue
            # all items interacted by the user
            all_index = np.where(users == target_user)[0]
            all_items = items[all_index]
            all_items = np.unique(all_items)
            same_target_items.append(all_items)

        self.same_target_users = same_target_users
        self.same_target_items = same_target_items

        mean_user = np.mean([len(users) for users in same_target_users[1:]])
        mean_item = np.mean([len(items) for items in same_target_items[1:]])
        # print("Mean same target user num", mean_user)
        # print("Mean same target item num", mean_item)
        self.user_same_target_mean_num = mean_user
        self.item_same_target_mean_num = mean_item


        return self.same_target_users, self.same_target_items


    if recbole.__version__ >= "1.1.1":

        def save(self):
            """Saving this :class:`Dataset` object to :attr:`config['checkpoint_dir']`."""
            save_dir = self.config["checkpoint_dir"]
            ensure_dir(save_dir)
            file = os.path.join(save_dir, f'{self.config["dataset"]}-{self.__class__.__name__}.pth')
            self.logger.info(
                set_color("Saving filtered dataset into ", "pink") + f"[{file}]"
            )
            with open(file, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def edge_index_to_adj_t(edge_index, edge_weight, m_num_nodes, n_num_nodes):
        adj = SparseTensor(row=edge_index[0],
                           col=edge_index[1],
                           value=edge_weight,
                           sparse_sizes=(m_num_nodes, n_num_nodes))
        return adj.t()


    def get_norm_adj_mat(self, enable_sparse=False):
        self.is_sparse = is_sparse
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.user_num + self.item_num

        if enable_sparse:
            if not is_sparse:
                self.logger.warning(
                    "Import `torch_sparse` error, please install corrsponding version of `torch_sparse`. Now we will use dense edge_index instead of SparseTensor in dataset.")
            else:
                adj_t = self.edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
                adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=False)
                return adj_t, None

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)

        return edge_index, edge_weight

    def get_bipartite_inter_mat(self, row='user', row_norm=True):
        r"""Get the row-normalized bipartite interaction matrix of users and items.
        """
        if row == 'user':
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row = self.inter_feat[row_field]
        col = self.inter_feat[col_field]
        edge_index = torch.stack([row, col])

        if row_norm:
            deg = degree(edge_index[0], self.num(row_field))
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            row_deg = degree(edge_index[0], self.num(row_field))
            col_deg = degree(edge_index[1], self.num(col_field))

            row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
            col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

            edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

        return edge_index, edge_weight

