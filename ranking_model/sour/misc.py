import numpy as np
import lightgbm as lgb


def prepare_lightgbm_sets(train_set, eval_set=None, include_train=False):
    train_set = lgb.Dataset(train_set[0], train_set[1], group=train_set[2], params={"name" : "train"})

    valid_sets = []
    valid_names = []
    if include_train:
        valid_sets = [train_set]
        valid_names = ["train"]

    if eval_set is not None:
        valid_sets += [lgb.Dataset(ds[0], ds[1], group=ds[2], reference=train_set, params={'name' : ds[3]}) for ds in eval_set]
        valid_names += [ds[3] for ds in eval_set]

    return train_set, valid_sets, valid_names

def remove_docs(data, labels, query_lens, idx_to_remove):
    idx_to_keep = np.setdiff1d(np.arange(data.shape[0]), idx_to_remove)
    qs_lens = np.repeat(np.arange(len(query_lens)), query_lens)
    new_qs_lens = np.bincount(qs_lens[idx_to_keep])

    return data[idx_to_keep], labels[idx_to_keep], new_qs_lens

def rename_dict_key(input_dict, key, key_alias):
    clean_dict = {key : value for key, value in input_dict.items()}
    key_to_remove = key_alias[np.argmax([k in clean_dict for k in clean_dict])]
    if key_to_remove in clean_dict:
        clean_dict[key] = clean_dict.pop(key_to_remove)
    return clean_dict