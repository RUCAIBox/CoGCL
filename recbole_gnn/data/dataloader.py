import numpy as np
import torch
from recbole.data.interaction import cat_interactions
from recbole.data.dataloader.general_dataloader import TrainDataLoader, NegSampleEvalDataLoader, FullSortEvalDataLoader

from recbole_gnn.data.transform import gnn_construct_transform


class CustomizedTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)

        # print("==================",self.transform)

    def collate_fn(self, index):

        index = np.array(index)
        data = self._dataset[index]
        data = self._neg_sampling(data)
        transformed_data = self.transform(self._dataset, data)
        return transformed_data

class CustomizedNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None


class CustomizedFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        if config['gnn_transform'] is not None:
            self.transform = gnn_construct_transform(config)




    def collate_fn(self, index):
        index = np.array(index)
        if not self.is_sequential:
            user_df = self.user_df[index]
            uid_list = list(user_df[self.uid_field])

            history_item = self.uid2history_item[uid_list]
            positive_item = self.uid2positive_item[uid_list]

            history_u = torch.cat(
                [
                    torch.full_like(hist_iid, i)
                    for i, hist_iid in enumerate(history_item)
                ]
            )
            history_i = torch.cat(list(history_item))

            positive_u = torch.cat(
                [torch.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_item)]
            )
            positive_i = torch.cat(list(positive_item))

            return self.transform(self._dataset, user_df), (history_u, history_i), positive_u, positive_i
        else:
            interaction = self._dataset[index]
            transformed_interaction = self.transform(self._dataset, interaction)
            inter_num = len(transformed_interaction)
            positive_u = torch.arange(inter_num)
            positive_i = transformed_interaction[self.iid_field]

            return transformed_interaction, None, positive_u, positive_i
