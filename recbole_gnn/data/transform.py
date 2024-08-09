import random
import time
from logging import getLogger

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from recbole.data.interaction import Interaction


def gnn_construct_transform(config):
    if config['gnn_transform'] is None:
        raise ValueError('config["gnn_transform"] is None but trying to construct transform.')
    str2transform = {
        'sess_graph': SessionGraph,
        "discrete_code": DiscreteCodeTransform,
        "code_positive": CodePositveSampling
    }
    return str2transform[config['gnn_transform']](config)



class CodePositveSampling:

    def __init__(self, config):
        self.config = config
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.sim_cl_weight = config['sim_cl_weight']


        self.AUG_USER_ID  = "aug_" + self.USER_ID
        self.AUG_ITEM_ID = "aug_" + self.ITEM_ID


    def __call__(self, dataset, interaction):


        if not hasattr(dataset, "code_similar_users") or not hasattr(dataset, "code_similar_items"):
            return interaction

        if dataset.code_similar_users is None or dataset.code_similar_items is None:
            return interaction

        if dataset.same_target_users is None or dataset.same_target_items is None:
            return interaction

        if self.sim_cl_weight <= 0:
            return interaction

        if self.USER_ID in interaction and self.ITEM_ID in interaction:

            same_target_users, same_target_items = dataset.get_same_targe_users_items()

            # print(dataset.code_similar_users[10][:3])


            users = interaction[self.USER_ID]
            items = interaction[self.ITEM_ID]

            device = users.device
            users = users.cpu().numpy()
            items = items.cpu().numpy()

            aug_users = []
            aug_items = []
            for i, (user_id, item_id) in enumerate(zip(users, items)):
                same_item_users = same_target_users[item_id]
                similar_users = dataset.code_similar_users[user_id]
                sim_num = len(same_item_users)//2 + 1
                # sim_num = len(same_item_users)
                if len(similar_users) > sim_num:
                    similar_users = np.random.choice(similar_users, size=sim_num , replace=False)
                cand_users = list(set(same_item_users) | set(similar_users))
                # cand_users = list(set(same_item_users))
                # cand_users = list(set(similar_users))
                if len(cand_users) == 1:
                    aug_users.append(cand_users[0])
                else:
                    sample_user = random.choice(cand_users)
                    while sample_user == user_id:
                        sample_user = random.choice(cand_users)
                    aug_users.append(sample_user)

                same_user_items = same_target_items[user_id]
                similar_items = dataset.code_similar_items[item_id]
                sim_num = len(same_user_items)//2 + 1
                # sim_num = len(same_user_items)
                if len(similar_items) > sim_num:
                    similar_items = np.random.choice(similar_items, size=sim_num, replace=False)
                cand_items = list(set(same_user_items) | set(similar_items))
                # cand_items = list(set(same_user_items))
                # cand_items = list(set(similar_items))
                if len(cand_items) == 1:
                    aug_items.append(cand_items[0])
                else:
                    sample_item = random.choice(cand_items)
                    while sample_item == item_id:
                        sample_item = random.choice(cand_items)
                    aug_items.append(sample_item)


            # default sorted
            unique_users_idx = np.unique(users, return_index=True)[1]
            unique_items_idx = np.unique(items, return_index=True)[1]

            aug_users = torch.LongTensor(aug_users)[unique_users_idx]
            aug_items = torch.LongTensor(aug_items)[unique_items_idx]

            new_dict = {
                self.AUG_USER_ID: aug_users,
                self.AUG_ITEM_ID: aug_items
            }
            interaction.update(Interaction(new_dict).to(device))



        return interaction




class DiscreteCodeTransform:
    def __init__(self, config):
        self.config = config
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID


        self.code_suffix = config["code_suffix"]
        self.USER_CODE = self.USER_ID + self.code_suffix
        self.ITEM_CODE = self.ITEM_ID + self.code_suffix
        self.NEG_ITEM_CODE = self.NEG_ITEM_ID + self.code_suffix

        self.AUG_USER_ID  = "aug_" + self.USER_ID
        self.AUG_ITEM_ID = "aug_" + self.ITEM_ID

        self.co_users = None
        self.co_items = None


    def get_co_users_items(self, dataset):
        if self.co_users is not None and self.co_items is not None:
            return self.co_users, self.co_items

        # print("Constructing co-users and co-items")
        co_users = []
        co_items = []
        users = dataset.inter_feat[self.USER_ID].numpy()
        items = dataset.inter_feat[self.ITEM_ID].numpy()

        for item_id in range(dataset.item_num):
            if item_id == 0:
                co_users.append([])
                continue
            # all users who interact with the item
            all_index = np.where(items == item_id)[0]
            all_users = users[all_index]
            all_users = np.unique(all_users)
            co_users.append(all_users)


        for user_id in range(dataset.user_num):
            if user_id == 0:
                co_items.append([])
                continue
            # all items interacted by the user
            all_index = np.where(users == user_id)[0]
            all_items = items[all_index]
            all_items = np.unique(all_items)
            co_items.append(all_items)

        self.co_users = co_users
        self.co_items = co_items

        return self.co_users, self.co_items



    def __call__(self, dataset, interaction):


        users = interaction[self.USER_ID]

        device = users.device
        users = users.cpu().numpy()

        user_codes = []
        for u in users:
            user_codes.append(dataset.user_codes[u].tolist())
        user_codes = torch.tensor(user_codes, dtype=torch.long, device=device) + dataset.user_num - 1
        new_dict = {
            self.USER_CODE: user_codes,
        }

        if self.ITEM_ID in interaction:
            pos_items = interaction[self.ITEM_ID]
            pos_items = pos_items.cpu().numpy()
            # print(pos_item)
            pos_item_codes = []
            for p in pos_items:
                pos_item_codes.append(dataset.item_codes[p].tolist())
            pos_item_codes = torch.tensor(pos_item_codes, dtype=torch.long, device=device) + dataset.item_num - 1
            new_dict[self.ITEM_CODE] = pos_item_codes

        if self.NEG_ITEM_ID in interaction:
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items = neg_items.cpu().numpy()
            neg_item_codes = []
            for n in neg_items:
                neg_item_codes.append(dataset.item_codes[n].tolist())
            neg_item_codes = torch.tensor(neg_item_codes, dtype=torch.long, device=device) + dataset.item_num - 1
            new_dict[self.NEG_ITEM_CODE] = neg_item_codes


        # print(self.config["model"])
        if self.USER_ID in interaction and self.ITEM_ID in interaction and self.config["model"] == "DuoGCL":
            co_users, co_items = self.get_co_users_items(dataset)
            # users = interaction[self.USER_ID].cpu().numpy()


            items = pos_items

            aug_users = []
            for i, item_id in enumerate(items):
                cand_users = co_users[item_id]
                if len(cand_users) == 1:
                    aug_users.append(cand_users[0])
                    continue

                sample_user = random.choice(cand_users)
                while sample_user == users[i]:
                    sample_user = random.choice(cand_users)
                aug_users.append(sample_user)

            aug_items = []
            for i, user_id in enumerate(users):
                cand_items = co_items[user_id]
                if len(cand_items) == 1:
                    aug_items.append(cand_items[0])
                    continue

                sample_item = random.choice(cand_items)
                while sample_item == items[i]:
                    sample_item = random.choice(cand_items)
                aug_items.append(sample_item)

            # default sorted
            unique_users_idx = np.unique(users, return_index=True)[1]
            unique_items_idx = np.unique(items, return_index=True)[1]

            aug_users = torch.LongTensor(aug_users)[unique_users_idx]
            aug_items = torch.LongTensor(aug_items)[unique_items_idx]

            new_dict[self.AUG_USER_ID] = aug_users.to(device)
            new_dict[self.AUG_ITEM_ID] = aug_items.to(device)

        interaction.update(Interaction(new_dict))
        return interaction

# class DiscreteCodeTransform:
#     def __init__(self, config):
#         self.USER_ID = config["USER_ID_FIELD"]
#         self.ITEM_ID = config["ITEM_ID_FIELD"]
#         self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
#
#     def __call__(self, dataset, interaction):
#
#         user = interaction[self.USER_ID]
#
#         device = user.device
#         user = user.cpu().numpy()
#
#         new_users = []
#
#         for u in user:
#             new_users.append(dataset.user_codes[u].tolist())
#
#         new_users = torch.tensor(new_users, dtype=torch.long, device=device)
#
#         new_dict = {
#             self.USER_ID: new_users,
#         }
#
#         try:
#             pos_item = interaction[self.ITEM_ID]
#             pos_item = pos_item.cpu().numpy()
#             new_pos_items = []
#             for p in pos_item:
#                 new_pos_items.append(dataset.item_codes[p].tolist())
#             new_pos_items = torch.tensor(new_pos_items, dtype=torch.long, device=device)
#             new_dict[self.ITEM_ID] = new_pos_items
#         except KeyError:
#             pass
#
#         try:
#             neg_item = interaction[self.NEG_ITEM_ID]
#             neg_item = neg_item.cpu().numpy()
#             new_neg_items = []
#             for n in neg_item:
#                 new_neg_items.append(dataset.item_codes[n].tolist())
#             new_neg_items = torch.tensor(new_neg_items, dtype=torch.long, device=device)
#             new_dict[self.NEG_ITEM_ID] = new_neg_items
#         except KeyError:
#             pass
#
#         interaction.update(Interaction(new_dict))
#         return interaction

class SessionGraph:
    def __init__(self, config):
        self.logger = getLogger()
        self.logger.info('SessionGraph Transform in DataLoader.')

    def __call__(self, dataset, interaction):
        graph_objs = dataset.graph_objs
        index = interaction['graph_idx']
        graph_batch = {
            k: [graph_objs[k][_.item()] for _ in index]
            for k in graph_objs
        }
        graph_batch['batch'] = []

        tot_node_num = torch.ones([1], dtype=torch.long)
        for i in range(index.shape[0]):
            for k in graph_batch:
                if 'edge_index' in k:
                    graph_batch[k][i] = graph_batch[k][i] + tot_node_num
            if 'alias_inputs' in graph_batch:
                graph_batch['alias_inputs'][i] = graph_batch['alias_inputs'][i] + tot_node_num
            graph_batch['batch'].append(torch.full_like(graph_batch['x'][i], i))
            tot_node_num += graph_batch['x'][i].shape[0]

        if hasattr(dataset, 'node_attr'):
            node_attr = ['batch'] + dataset.node_attr
        else:
            node_attr = ['x', 'batch']
        for k in node_attr:
            graph_batch[k] = [torch.zeros([1], dtype=graph_batch[k][-1].dtype)] + graph_batch[k]

        for k in graph_batch:
            if k == 'alias_inputs':
                graph_batch[k] = pad_sequence(graph_batch[k], batch_first=True)
            else:
                graph_batch[k] = torch.cat(graph_batch[k], dim=-1)

        interaction.update(Interaction(graph_batch))
        return interaction
