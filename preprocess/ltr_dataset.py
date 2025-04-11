import torch
from torch.utils.data import Dataset
from sklearn.exceptions import NotFittedError

class LearningToRankDataset(Dataset):
    def __init__(self, df, label_column, list_id_column, variables, transform=None, seed=None, device=None, pnl_column=None):
        # It is costly to sort before any filtering happens, but we need the groups to be together for later efficiency.
        # All later steps are expected to maintain query group order.
        df.sort_values(by=list_id_column, inplace=True)
        feat_columns = variables
        self.feat = df[feat_columns].values
        if transform is not None:
            try:
                self.feat = transform.transform(self.feat)
            except:
                self.feat = transform.fit_transform(self.feat)

        self.feat = torch.from_numpy(self.feat).float().to(device)
        self.target = torch.from_numpy(df[label_column].values).float().to(device)
        self.length = torch.from_numpy(df[list_id_column].value_counts(sort=False).values).to(device)
        self.cum_length = torch.cumsum(self.length, dim=0).to(device)

        if pnl_column is not None:
            self.pnl = torch.from_numpy(df[pnl_column].values).float().to(device)
        else:
            self.pnl = None

        if 'explicit_target' in df.columns:
            self.explicit_target = torch.from_numpy(df['explicit_target'].values).int().to(device)
        else:
            self.explicit_target = None

    def __getitem__(self, item):
        # All item features, targets and list ids are stored in a single flat array. Each list is stored back-to-back.
        # When getting a batch element (i.e. a list), we therefore need to slice the correct range in the flat array.
        # The start and end indices of each list can be inferred from the cum_length array.

        if item == 0:
            start_idx = 0
        else:
            start_idx = self.cum_length[item-1]
        end_idx = self.cum_length[item].item()

        item_dict = {
            'feat': self.feat[start_idx:end_idx],
            'target': self.target[start_idx:end_idx],
            'length': self.length[item].reshape(1),
        }

        if self.pnl is not None:
            item_dict['pnl'] = self.pnl[start_idx:end_idx]

        return item_dict

    def __len__(self):
        return self.length.shape[0]

    @staticmethod
    def collate_fn(batches):
        batch_example = batches[0]
        batch = {key: torch.cat([batch_vals[key] for batch_vals in batches]) for key in batch_example.keys()}
        return batch

    @property
    def input_dim(self):
        return self.feat.shape[1]

    @property
    def max_target(self):
        # Used in the ordinal loss function of the RankFormer
        return self.target.max().cpu().int().item()


class EmbedDataset(Dataset):
    def __init__(self, df, label_column, list_id_column, cat_variables, conti_variables, transform=None, seed=None, device=None, pnl_column=None):
        # It is costly to sort before any filtering happens, but we need the groups to be together for later efficiency.
        # All later steps are expected to maintain query group order.
        df.sort_values(by=list_id_column, inplace=True)
        cat_feat_columns = cat_variables
        conti_feat_columns = conti_variables
        self.cat_feat_columns = df[cat_feat_columns].values
        self.conti_feat_columns = df[conti_feat_columns].values

        self.num_continuous = len(conti_variables)

        if transform is not None:
            try:
                self.conti_feat_columns = transform.transform(self.conti_feat_columns)
            except:
                self.conti_feat_columns = transform.fit_transform(self.conti_feat_columns)

        self.cat_feat_columns = torch.from_numpy(self.cat_feat_columns).long().to(device)
        self.conti_feat_columns = torch.from_numpy(self.conti_feat_columns).float().to(device)
        self.target = torch.from_numpy(df[label_column].values).float().to(device)
        self.length = torch.from_numpy(df[list_id_column].value_counts(sort=False).values).to(device)
        self.cum_length = torch.cumsum(self.length, dim=0).to(device)
        self.categ_max_classes = tuple([df[col].max() for col in cat_feat_columns])
        if pnl_column is not None:
            self.pnl = torch.from_numpy(df[pnl_column].values).float().to(device)
        else:
            self.pnl = None

    def __getitem__(self, item):
        # All item features, targets and list ids are stored in a single flat array. Each list is stored back-to-back.
        # When getting a batch element (i.e. a list), we therefore need to slice the correct range in the flat array.
        # The start and end indices of each list can be inferred from the cum_length array.

        if item == 0:
            start_idx = 0
        else:
            start_idx = self.cum_length[item-1]
        end_idx = self.cum_length[item].item()

        item_dict = {
            'cat_feat': self.cat_feat_columns[start_idx:end_idx],
            'conti_feat': self.conti_feat_columns[start_idx:end_idx],
            'target': self.target[start_idx:end_idx],
            'length': self.length[item].reshape(1),
        }

        if self.pnl is not None:
            item_dict['pnl'] = self.pnl[start_idx:end_idx]

        return item_dict

    def __len__(self):
        return self.length.shape[0]

    @staticmethod
    def collate_fn(batches):
        batch_example = batches[0]
        batch = {key: torch.cat([batch_vals[key] for batch_vals in batches]) for key in batch_example.keys()}
        return batch

    @property
    def input_dim(self):
        return self.conti_feat_columns.shape[1]

    @property
    def max_target(self):
        # Used in the ordinal loss function of the RankFormer
        return self.target.max().cpu().int().item()

    @property
    def max_cat_classes(self):
        return self.categ_max_classes
