import os
from datetime import datetime
import lightgbm as lgb
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from scipy.special import expit  # Sigmoid function

import util

# Global variable to store group information for training
GLOBAL_GROUPS = None

GLOBAL_GROUPS = None
GLOBAL_PROFITS = None

def setup_globals(train_set, group_column, profit_column):
    global GLOBAL_GROUPS, GLOBAL_PROFITS
    GLOBAL_GROUPS = train_set.groupby(group_column).size().to_numpy()
    GLOBAL_PROFITS = train_set[profit_column].values


def pa_bce_objective(y_true, y_pred, weight):
    global GLOBAL_GROUPS, GLOBAL_PROFITS
    top_k = 20  # Predefined value inside the function

    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    group_ptr = np.cumsum(np.append([0], GLOBAL_GROUPS))

    for idx in range(len(GLOBAL_GROUPS)):
        start, end = group_ptr[idx], group_ptr[idx + 1]

        group_preds = y_pred[start:end]
        group_profits = GLOBAL_PROFITS[start:end]
        group_labels = y_true[start:end]

        # Sort by true profit to select top-K instances
        sorted_indices = np.argsort(-group_profits)
        top_k_indices = sorted_indices[:min(top_k, len(sorted_indices))]

        group_preds = group_preds[top_k_indices]
        group_profits = group_profits[top_k_indices]
        group_labels = group_labels[top_k_indices]

        pred_diff = group_preds[:, None] - group_preds[None, :]
        profit_diff = group_profits[:, None] - group_profits[None, :]

        # Compute loss only for upper triangular matrix (i < j)
        mask = np.triu(np.ones_like(pred_diff, dtype=bool), k=1)

        pred_prob = expit(pred_diff)

        # see when to use log, only when the magnitude of profit_diff is large

        pnl_gap = np.log1p(np.abs(profit_diff))
        S_ij = (profit_diff > 0).astype(float)

        lambda_ij = pnl_gap * (pred_prob - S_ij)
        hess_ij = pnl_gap * pred_prob * (1 - pred_prob)

        # Aggregate results back to the correct top_k indices in grad/hess
        grad[start:end][top_k_indices] = np.sum(lambda_ij, axis=1)
        hess[start:end][top_k_indices] = np.sum(hess_ij, axis=1)

    return grad, hess



def run_lambdamart(args):
    global GLOBAL_GROUPS  # use global variable to store group info

    if args.dataset == 'creditcard':
        variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15",
                     "V16",
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

    elif args.dataset == 'jobprofit':
        # bedrooms,bathrooms,sqft_living,floors,waterfront,view,condition
        variables_num = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs',
                         'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales',
                         'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours',
                         'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs',
                         'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed',
                         'Materials__Equipment__PO_Costs']
        variables_cat = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
        variables = variables_num + variables_cat

    else:
        raise Exception('Dataset not implemented')

    # Load data and sort by query id.
    label_column = 'Class'
    train_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', '100', 'train.csv'))
    train_set = train_set.sort_values(by=['qid'], ascending=[True])
    val_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', '100', 'val.csv'))
    val_set = val_set.sort_values(by=['qid'], ascending=[True])

    X_train = train_set[variables + ['qid']]
    y_train = train_set[label_column]
    X_val = val_set[variables + ['qid']]
    y_val = val_set[label_column]
    qids_train = X_train.groupby("qid")["qid"].count().to_numpy()
    qids_valid = X_val.groupby("qid")["qid"].count().to_numpy()

    # Setup global variables for custom PA-BCE objective
    setup_globals(train_set, 'qid', 'Amount')

    # Store training groups globally for use in the custom objective.
    GLOBAL_GROUPS = qids_train

    # Instantiate the ranker.
    if args.objective == 'pabce':
        # Use our custom PAG-BCE objective.
        ranker = lgb.LGBMRanker(objective=pa_bce_objective, metric='ndcg', n_estimators=args.n_estimators)
    elif args.objective == 'lambdarank':
        ranker = lgb.LGBMRanker(objective="lambdarank", metric='ndcg', n_estimators=args.n_estimators)
    else:
        raise Exception("Unknown objective {}".format(args.objective))

    ranker.fit(X_train, y_train, group=qids_train,
               eval_set=[(X_val, y_val)], eval_group=[qids_valid],
               eval_at=[1, 2, 4, 6, 8, 10, 25, 50])

    # lgb.plot_importance(ranker, importance_type="gain", figsize=(10, 6),
    #                     title="LightGBM Feature Importance (Gain)")
    # plt.show()

    # Generate predictions and ranking labels.
    train_set['fst_step_scores'] = ranker.predict(X_train)
    train_set['ranking_label'] = train_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    val_set['fst_step_scores'] = ranker.predict(X_val)
    val_set['ranking_label'] = val_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    test_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', '100', 'test.csv'))
    test_set = test_set.sort_values(by=['qid'], ascending=[True])
    X_test = test_set[variables + ['qid']]
    ranking_score = ranker.predict(X_test)
    test_set['fst_step_scores'] = ranking_score
    test_set['ranking_label'] = test_set.groupby('qid')['fst_step_scores'].rank(method='first', ascending=False)

    # Save results.
    model_path = os.path.join('storage', args.dataset, f'lambdamart_benchmark_{args.n_estimators}_fold{args.fold}') if args.objective == 'lambdarank' else os.path.join('storage', args.dataset, f'lambdamart_pabce_{args.n_estimators}_fold{args.fold}')
    os.makedirs(model_path, exist_ok=True)
    train_set.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(model_path, 'train.csv'), index=False)
    val_set.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(model_path, 'val.csv'), index=False)
    test_set.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(model_path, 'test.csv'), index=False)

    train_ndcg_3, train_ndcg_5, train_ndcg_10, train_mrr = util.compute_ranking_metrics(train_set)

    test_ndcg_3, test_ndcg_5, test_ndcg_10, test_mrr = util.compute_ranking_metrics(test_set)

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
    parser.add_argument("--group_size", type=int, choices=[20, 30, 50, 100, 200],
                        required=True)
    parser.add_argument("--strategy", type=str, choices=['ordinal', 'binary'],
                        required=True)
    parser.add_argument("--objective", type=str, choices=['lambdarank', 'pabce'],
                        required=True)
    parser.add_argument("--n_estimators", type=int, default=10000)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    run_lambdamart(args)
