from torch.utils.data import DataLoader, Dataset
import torch


class FTDataset(Dataset):
    def __init__(self, df, label_column, conti_variables, cat_variables, transform=None, seed=None, device=None):
        # It is costly to sort before any filtering happens, but we need the groups to be together for later efficiency.
        # All later steps are expected to maintain query group order.
        self.conti_feat = df[conti_variables].values
        self.categ_feat = df[cat_variables].values

        self.num_continuous = len(conti_variables)

        if transform is not None:
            try:
                self.conti_feat = transform.transform(self.conti_feat)
            except:
                self.conti_feat = transform.fit_transform(self.conti_feat)
        self.conti_feat = torch.from_numpy(self.conti_feat).float()
        self.categ_feat = torch.from_numpy(self.categ_feat).long()
        self.target = torch.from_numpy(df[label_column].values).float()

        self.categ_max_classes = tuple([df[col].max() for col in cat_variables])

    def __getitem__(self, idx):
        # All item features, targets and list ids are stored in a single flat array. Each list is stored back-to-back.
        # When getting a batch element (i.e. a list), we therefore need to slice the correct range in the flat array.
        # The start and end indices of each list can be inferred from the cum_length array.

        item_dict = {
            'conti_feat': self.conti_feat[idx],
            'cat_feat': self.categ_feat[idx],
            'target': self.target[idx]
        }

        return item_dict

    def __len__(self):
        return self.conti_feat.shape[0]

    @property
    def max_cat_classes(self):
        return self.categ_max_classes