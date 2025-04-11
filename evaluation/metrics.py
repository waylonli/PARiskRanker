from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate(df, with_prior=True, dev_set=None):
    # calculate F1 score, AUC score, sensitivity, specificity, precision

    if not with_prior:
        if df['fst_step_scores'].max() <= 1 and df['fst_step_scores'].min() >= 0:
            optimal_threshold = tune_threshold_on_dev_set(dev_set) if dev_set is not None else 0.5
            df['pred'] = [1 if y >= optimal_threshold else 0 for y in df['pred_proba']]
        y_pred = df['pred'].values
    else:
        # the prior is top-1% anomaly rate
        # sort the pred_proba in descending order and take the top 1%
        df = df.sort_values(by='pred_proba', ascending=False)
        df['new_pred'] = 0
        df.iloc[:int(len(df) * 0.01), df.columns.get_loc('new_pred')] = 1
        y_pred = df['new_pred'].values

    y_true = df['Class'].values
    y_proba_1 = df['pred_proba'].values
    y_proba_0 = 1 - y_proba_1
    # keep 4 decimal places
    cm = confusion_matrix(y_true, y_pred)
    specificity = round(cm[0, 0] / (cm[0, 0] + cm[0, 1]), 4)
    sensitive = round(cm[1, 1] / (cm[1, 0] + cm[1, 1]), 4)
    # Calculate P&L
    PnL = round(sum(-np.array(df['Amount']) * np.abs(np.array(y_true) - np.array(y_pred))), 4)
    eval_dict = dict()
    eval_dict['f1_score'] = round(f1_score(y_true, y_pred, average='macro'), 4)
    eval_dict['PnL'] = PnL
    eval_dict['auc_score'] = round(roc_auc_score(y_true, y_proba_1), 4)
    eval_dict['precision'] = round(precision_score(y_true, y_pred, average='binary'), 4)
    eval_dict['sensitivity'] = sensitive
    eval_dict['specificity'] = specificity
    # check if there's negative profit

    # try:
    #     # check if there is negative profit
    #     eval_dict['anomaly_profit'] = df[(df['Class'] == 1) & (df['new_pred'] == 1)]['Amount']
    #     eval_dict['true_anomaly_profit'] = df[(df['Class'] == 1)]['Amount']
    # except:
    #     eval_dict['anomaly_profit'] = df[(df['Class'] == 1) & (df['pred'] == 1)]['Amount']
    #     eval_dict['true_anomaly_profit'] = df[(df['Class'] == 1)]['Amount']
    return eval_dict

def tune_threshold_on_dev_set(dev_set):
    y_proba_1 = dev_set['pred_proba'].values
    y_true = dev_set['Class'].values
    # sort the pred_proba in descending order
    dev_set = dev_set.sort_values(by='pred_proba', ascending=False)
    # try different thresholds
    thresholds = np.arange(1.0, 0.0, -0.01)
    optimal_threshold = 0
    optimal_f1 = 0
    for t in thresholds:
        y_pred = [1 if y >= t else 0 for y in y_proba_1]
        # make sure there is at least one positive prediction
        if sum(y_pred) == 0:
            continue
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > optimal_f1:
            optimal_f1 = f1
            optimal_threshold = t
    return optimal_threshold


