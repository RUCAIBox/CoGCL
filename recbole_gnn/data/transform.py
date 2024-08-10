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

                if len(similar_users) > sim_num:
                    similar_users = np.random.choice(similar_users, size=sim_num , replace=False)
                cand_users = list(set(same_item_users) | set(similar_users))

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

                if len(similar_items) > sim_num:
                    similar_items = np.random.choice(similar_items, size=sim_num, replace=False)
                cand_items = list(set(same_user_items) | set(similar_items))

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



