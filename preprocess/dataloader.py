from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader
from preprocess.ltr_dataset import LearningToRankDataset, EmbedDataset

def load_data(df, stage, variables, batch_size=64, seed=42, num_workers=0, device='cpu', label_column='target', pnl_column=None):

    # transform = QuantileTransformer(output_distribution='normal')
    transform = StandardScaler()

    ltr_data = LearningToRankDataset(df,
                                     label_column=label_column,
                                     list_id_column='qid',
                                     variables=variables,
                                     transform=transform,
                                     seed=seed,
                                     device=device,
                                     pnl_column=pnl_column)

    if stage == 'train':
        train_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=True, collate_fn=LearningToRankDataset.collate_fn,
                                num_workers=num_workers)
        return train_loader, ltr_data.input_dim, ltr_data.max_target
    else:
        test_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=num_workers)
        return test_loader, ltr_data.input_dim, ltr_data.max_target

def load_data_ft(df, stage, cat_variables, conti_variables, batch_size=64, seed=42, num_workers=0, device='cpu', label_column='target', pnl_column=None):

    # transform = QuantileTransformer(output_distribution='normal')
    transform = StandardScaler()
    ltr_data = EmbedDataset(df,
                             label_column=label_column,
                             list_id_column='qid',
                             cat_variables=cat_variables,
                             conti_variables=conti_variables,
                             transform=transform,
                             seed=seed,
                             device=device,
                             pnl_column=pnl_column)

    if stage == 'train':
        train_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=True, collate_fn=LearningToRankDataset.collate_fn,
                                num_workers=num_workers)
        return train_loader, ltr_data.input_dim, ltr_data.max_target
    else:
        test_loader = DataLoader(ltr_data, batch_size=batch_size, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=num_workers)
        return test_loader, ltr_data.input_dim, ltr_data.max_target

