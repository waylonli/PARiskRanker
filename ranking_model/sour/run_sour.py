import os
from datetime import datetime
from ranking_model.sour.model import SOUR
import argparse
import lightgbm as lgb
import numpy as np
import pandas as pd
# import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# wandb offline
# wandb.offline = True

def run_sour(args):

    current_time = datetime.now().strftime('%b%d_%H:%M:%S')
    # init wandb
    # wandb.init(project="Risky-Trader-Prediction",
    #            entity="uoe-turing",
    #            name="Run-SOUR-{}-{}".format(args.strategy, current_time),
    #            tags=['ranking', 'train', 'test', str(args.group_size), args.strategy],
    #            config=vars(args))

    # define variables
    variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                 "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

    # Load data to data loader
    label_column = 'Class'
    train_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'train.csv'))
    train_set = train_set.sort_values(by=['qid'], ascending=[True])
    val_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'val.csv'))
    val_set = val_set.sort_values(by=['qid'], ascending=[True])

    X_train = train_set[variables].values
    y_train = train_set[label_column]
    X_val = val_set[variables].values
    y_val = val_set[label_column]
    qids_train = train_set.groupby("qid")["qid"].count().to_numpy()
    qids_valid = train_set.groupby("qid")["qid"].count().to_numpy()
    # train model
    sour_ranker = SOUR(queries=X_train, labels=y_train, qs_len=qids_train)
    params_dict = {
        "objective": "lambdarank",
        "metric": 'ndcg',
        "n_estimators": 1000,
        "eval_set": [(X_val, y_val)],
        "eval_group": [qids_valid],
        "eval_at": 50
    }
    # ranker = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=10, objective="rank:ndcg",
    #                        lambdarank_pair_method="topk")
    ranker = sour_ranker.train(params_dict, outliers_type="all", start=0, end=100)
    # ranker.fit(X_train, y_train)
    lgb.plot_importance(ranker, importance_type="gain", figsize=(10, 6), title="LightGBM Feature Importance (Gain)")
    plt.show()
    # generate ranking score for train set
    train_set['fst_step_scores'] = ranker.predict(X_train)
    train_set['ranking_label'] = train_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    # generate ranking score for val set
    val_set['fst_step_scores'] = ranker.predict(X_val)
    val_set['ranking_label'] = val_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    # generate ranking score for test set
    test_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'test.csv'))
    test_set = test_set.sort_values(by=['qid'], ascending=[True])
    X_test = test_set[variables].values
    ranking_score = ranker.predict(X_test)
    test_set['fst_step_scores'] = ranking_score
    test_set['ranking_label'] = test_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    # save results datasets
    # create folder if not exist
    model_path = os.path.join('storage', 'sour_{}_{}_fold{}'.format(args.group_size, args.strategy, str(args.fold)))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # shuffle the data
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    val_set = val_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    train_set.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(model_path, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(model_path, 'test.csv'), index=False)

    train_ndcg_3 = 0
    train_ndcg_5 = 0
    train_ndcg_10 = 0
    train_mrr = 0
    validate_train_qids = 0
    test_ndcg_3 = 0
    test_ndcg_5 = 0
    test_ndcg_10 = 0
    test_mrr = 0
    validate_test_qids = 0

    for qid in tqdm(train_set['qid'].unique(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    desc="Computing NDCG score for training set"):
        target_label = train_set[train_set['qid'] == qid]['Class'].values.astype(int)
        if max(target_label) == 0:
            continue
        validate_train_qids += 1
        rank_score = train_set[train_set['qid'] == qid]['fst_step_scores'].values
        if max(rank_score) > 1 or min(rank_score) < 0:
            rank_score = 1 / (1 + np.exp(-rank_score))
        train_ndcg_3 += ndcg_score([target_label], [rank_score], k=3)
        train_ndcg_5 += ndcg_score([target_label], [rank_score], k=5)
        train_ndcg_10 += ndcg_score([target_label], [rank_score], k=10)
        train_mrr += 1 / (np.where(rank_score == max(rank_score))[0][0] + 1)
    train_ndcg_3 /= validate_train_qids
    train_ndcg_5 /= validate_train_qids
    train_ndcg_10 /= validate_train_qids
    train_mrr /= validate_train_qids

    for qid in tqdm(test_set['qid'].unique(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    desc="Computing NDCG score for testing set"):
        target_label = test_set[test_set['qid'] == qid]['Class'].values.astype(int)
        if max(target_label) == 0:
            continue
        validate_test_qids += 1
        rank_score = test_set[test_set['qid'] == qid]['fst_step_scores'].values
        test_ndcg_3 += ndcg_score([target_label], [rank_score], k=3)
        test_ndcg_5 += ndcg_score([target_label], [rank_score], k=5)
        test_ndcg_10 += ndcg_score([target_label], [rank_score], k=10)
        test_mrr += 1 / (np.where(rank_score == max(rank_score))[0][0] + 1)
    test_ndcg_3 /= validate_test_qids
    test_ndcg_5 /= validate_test_qids
    test_ndcg_10 /= validate_test_qids
    test_mrr /= validate_test_qids

    # wandb.log(
    #     {'test_NDCG@3': test_ndcg_3, 'test_NDCG@5': test_ndcg_5, 'test_NDCG@10': test_ndcg_10, 'test_MRR': test_mrr})
    print("Train NDCG@3: {:.4f}".format(train_ndcg_3))
    print("Train NDCG@5: {:.4f}".format(train_ndcg_5))
    print("Train NDCG@10: {:.4f}".format(train_ndcg_10))
    print("Train MRR: {:.4f}".format(train_mrr))
    print("Test NDCG@3: {:.4f}".format(test_ndcg_3))
    print("Test NDCG@5: {:.4f}".format(test_ndcg_5))
    print("Test NDCG@10: {:.4f}".format(test_ndcg_10))
    print("Test MRR: {:.4f}".format(test_mrr))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_size", type=int, choices=[20, 30, 50, 100, 200], required=True, default=100)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--strategy", type=str, choices=['ordinal', 'binary'], required=True, default='ordinal')
    parser.add_argument("--dataset", type=str, default='creditcard')

    args = parser.parse_args()
    run_sour(args)