import argparse
import os
from deepod.models.tabular import DeepSAD, DeepIsolationForest, FeaWAD, SLAD
import pandas as pd
import pickle
import torch
from evaluation.metrics import evaluate
from deepod.metrics import tabular_metrics

def train_and_test_outlier(model):
    variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                 "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
    train_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/train.csv')
    val_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/val.csv')
    test_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/test.csv')
    X_train = train_set[variables].values
    y_train = train_set["Class"].values
    X_val = val_set[variables].values
    y_val = val_set["Class"].values
    X_test = test_set[variables].values
    y_test = test_set["Class"].values

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model == 'deepsad':
        model = DeepSAD(device=device,
                        epochs=200,
                        batch_size=64,
                        hidden_dims='100,50',
                        bias=True)
    elif model == 'deepisolationforest':
        model = DeepIsolationForest(device=device,
                                    epochs=200,
                                    n_estimators=100,
                                    max_samples=256,
                                    n_jobs=-1,
                                    batch_size=64)
    elif model == 'feawad':
        model = FeaWAD(device=device,
                       epochs=200,
                       batch_size=64,
                       lr=1e-3,
                       rep_dim=128,
                       margin=4.,
                       hidden_dims='100,50',)
    elif model == 'slad':
        model = SLAD(device=device,
                     epochs=200,
                     batch_size=128,
                     hidden_dims=256)
    else:
        raise Exception('Model not implemented')


    model.fit(X_train, y_train)

    # generate first step scores for train set
    train_set['fst_step_scores'] = model.decision_function(X_train)
    train_set['fst_step_pred'] = model.predict(X_train)

    # generate first step scores for val set
    val_set['fst_step_scores'] = model.decision_function(X_val)
    val_set['fst_step_pred'] = model.predict(X_val)

    # generate first step scores for test set
    test_set['fst_step_scores'] = model.decision_function(X_test)
    test_set['fst_step_pred'] = model.predict(X_test)

    model_path = os.path.join('storage', '{}_benchmark_fold{}'.format(args.model, str(args.fold)))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_set.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(model_path, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(model_path, 'test.csv'), index=False)
    # save model by pickle
    with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rosas')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    print("===== Running {} benchmark =====".format(args.model))
    train_and_test_outlier(args.model)