import pdb
from datetime import datetime

import numpy as np

import util
from preprocess.ltr_dataset import EmbedDataset
from preprocess.ft_dataset import FTDataset
from ranking_model.pariskranker import PARiskRanker
import argparse
import os
import pandas as pd
import json
import torch
from preprocess.dataloader import load_data_ft
from tqdm import tqdm
from sklearn.metrics import ndcg_score

def train_pa_riskranker(args):

    is_binary = True if args.strategy == 'binary' else False
    batch_size_map = {20: 512, 30: 512, 50: 512, 100: 256, 200: 256, 500: 128, 1000: 32}
    args.batch_size = batch_size_map[args.group_size]

    # Load data to data loader
    train_df = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'train.csv'))
    val_df = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'val.csv'))
    test_df = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'test.csv'))


    # exclude ['Amount', 'Class', 'qid']
    if args.dataset == 'creditcard':
        conti_variables = train_df.drop(columns=['Amount', 'Class', 'qid'], axis=1).columns.tolist()
        categ_variables = []
    elif args.dataset == "jobprofit":
        conti_variables = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs',
                         'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales',
                         'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours',
                         'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs',
                         'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed',
                         'Materials__Equipment__PO_Costs']
        categ_variables = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
    else:
        raise ValueError("Dataset not supported")

    label_col = 'Class'

    train_loader, _, max_target = load_data_ft(train_df, stage='train', label_column=label_col, cat_variables=categ_variables, conti_variables=conti_variables, batch_size=args.batch_size, pnl_column='Amount')
    dev_loader, _, _ = load_data_ft(val_df, stage='dev', label_column=label_col, cat_variables=categ_variables, conti_variables=conti_variables, batch_size=args.batch_size, pnl_column='Amount')

    # save the config
    store_path = "./storage/{}/pariskranker_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss_fn, args.fold)
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    if args.ft_embedding_checkpoint is None:
        ft_train_data = (FTDataset(train_df, label_column='Class', cat_variables=categ_variables, conti_variables=conti_variables,
                   device='cpu'),
        FTDataset(val_df, label_column='Class', cat_variables=categ_variables, conti_variables=conti_variables,
                   device='cpu'))
        ft_checkpoint = None
    elif not os.path.exists(args.ft_embedding_checkpoint):
        ft_train_data = (
            FTDataset(train_df, label_column='Class', cat_variables=categ_variables, conti_variables=conti_variables,
                      device='cpu'),
            FTDataset(val_df, label_column='Class', cat_variables=categ_variables, conti_variables=conti_variables,
                      device='cpu'))
        ft_checkpoint = None
    else:
        ft_train_data = None
        ft_checkpoint = args.ft_embedding_checkpoint

    # init model
    pa_riskranker = PARiskRanker(
        input_dim=32,
        tf_dim_feedforward=args.tf_dim_feedforward,
        tf_nhead=args.tf_nhead,
        tf_num_layers=args.tf_num_layers,
        head_hidden_layers=[32],
        dropout=args.dropout,
        output_embedding_mode=False,
        loss_fn=args.loss_fn,
        ft_embedder_train_data=ft_train_data,
        ft_embedder_checkpoint_path=ft_checkpoint,
        freeze_ft_embedder=args.freeze_ft_embedder,
        dataset_name=args.dataset,
    )
    
    # print trainable parameters sum in unit of million
    trainable_params = sum(p.numel() for p in pa_riskranker.parameters() if p.requires_grad)
    print("Trainable parameters: {:.2f}M".format(trainable_params / 1000000))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pa_riskranker.to(device)
    config_dict = dict()
    # add fc_model_config, transformer_config, post_model_config to config_dict
    config_dict['group_size'] = args.group_size
    config_dict['strategy'] = args.strategy
    with open(os.path.join(store_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     pa_riskranker = torch.nn.DataParallel(pa_riskranker)

    # init optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(pa_riskranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(pa_riskranker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")

    # init scheduler
    decay_fn = lambda t: (t - 10 + 1) ** -.05 if t >= 10 else 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_fn)

    # start training
    print("Start training...")
    best_dev_loss = float('inf')
    loop = tqdm(range(1, args.epochs+1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    since_last_improvement = 0

    for epoch in loop:
        log_loss = 0
        if epoch >= 100:
            since_last_improvement += 1
        pa_riskranker.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            cat_feat, conti_feat, length, target, pnl = batch['cat_feat'].to(device), batch['conti_feat'].to(device), \
            batch['length'].to(device), batch['target'].to(device), batch['pnl'].to(device)
            optimizer.zero_grad()
            y_pred = pa_riskranker(cat_feat, conti_feat, length)
            loss = pa_riskranker.compute_loss(y_pred, target, length, binary=is_binary) if not args.pnl \
                else pa_riskranker.compute_loss(y_pred, target, length, pnl=pnl, binary=is_binary)
            log_loss += loss.item()
            loss.backward()
            optimizer.step()

        loop.set_description(f"Epoch {epoch} | Train Loss {round(log_loss / len(train_loader), 4)}")
        # evaluate on dev set every 5 epochs and save the best model
        if epoch % 1 == 0:
            pa_riskranker.eval()
            dev_loss = 0
            for batch in dev_loader:
                cat_feat, conti_feat, length, target, pnl = batch['cat_feat'].to(device), batch['conti_feat'].to(
                    device), \
                    batch['length'].to(device), batch['target'].to(device), batch['pnl'].to(device)
                y_pred = pa_riskranker(cat_feat, conti_feat, length)
                loss = pa_riskranker.compute_loss(y_pred, target, length, binary=is_binary) if not args.pnl \
                    else pa_riskranker.compute_loss(y_pred, target, length, pnl=pnl, binary=is_binary)
                dev_loss += loss.item()
            dev_loss /= len(dev_loader)

            loop.set_description(f"Epoch {epoch} | Train Loss {round(log_loss / len(train_loader), 4)} | Dev Loss {round(dev_loss, 4)}")
            if dev_loss < best_dev_loss:
                since_last_improvement = 0
                print(f"Saving best model at epoch {epoch}")
                best_dev_loss = dev_loss
                if os.path.exists(os.path.join(store_path, 'best_model.pt')):
                    os.remove(os.path.join(store_path, 'best_model.pt'))
                torch.save(pa_riskranker, os.path.join(store_path, 'best_model.pt'))
                # save the config file
        # early stopping if the dev loss does not decrease for 100 epochs
        if since_last_improvement >= 40:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step()

    return

def test_pa_riskranker(args):

    model_path = os.path.join('storage', args.dataset, 'pariskranker_{}_{}_{}_fold{}'.format(args.model_group_size, args.strategy, args.loss_fn, args.fold))
    output_path = os.path.join(model_path, args.test_group_size)
    is_binary = True if args.strategy == 'binary' else False
    # load the best model and config
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pa_riskranker = torch.load(os.path.join(model_path, 'best_model.pt'))

    pa_riskranker.to(device)
    pa_riskranker.eval()

    train_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.test_group_size), 'train.csv')).sort_values(
        by=['qid', 'Class'])
    dev_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.test_group_size), 'val.csv')).sort_values(by=['qid', 'Class'])
    test_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.test_group_size), 'test.csv')).sort_values(
        by=['qid', 'Class'])

    # combine categorical columns, turns one hot encoding to integer encoding
    if args.dataset == 'creditcard':
        conti_variables = train_set.drop(columns=['Amount', 'Class', 'qid'], axis=1).columns.tolist()
        categ_variables = []
    elif args.dataset == "jobprofit":
        conti_variables = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs',
                         'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales',
                         'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours',
                         'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs',
                         'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed',
                         'Materials__Equipment__PO_Costs']
        categ_variables = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
    else:
        raise ValueError("Dataset not supported")

    label_col = 'Class'

    train_loader, _, max_target = load_data_ft(train_set, stage='train', label_column=label_col,
                                               cat_variables=categ_variables, conti_variables=conti_variables,
                                               batch_size=1)
    dev_loader, _, _ = load_data_ft(dev_set, stage='dev', label_column=label_col, cat_variables=categ_variables,
                                    conti_variables=conti_variables, batch_size=1)
    test_loader, _, _ = load_data_ft(test_set, stage='test', label_column=label_col, cat_variables=categ_variables,
                                     conti_variables=conti_variables, batch_size=1)


    train_ranking_score = []
    train_set_loop = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    train_set_loop.set_description("Computing ranking score for training set")
    for batch in train_set_loop:
        cat_feat, conti_feat, length, target = batch['cat_feat'].to(device), batch['conti_feat'].to(
            device), batch['length'].to(device), batch['target'].to(device)
        y_pred = pa_riskranker(cat_feat, conti_feat, length).squeeze()
        try:
            train_ranking_score.extend(y_pred.detach().cpu().numpy())
        except:
            train_ranking_score.append(y_pred.detach().cpu().numpy())

    train_set['fst_step_scores'] = train_ranking_score

    dev_ranking_score = []
    dev_set_loop = tqdm(dev_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    dev_set_loop.set_description("Computing ranking score for dev set")
    for batch in dev_set_loop:
        cat_feat, conti_feat, length, target = batch['cat_feat'].to(device), batch['conti_feat'].to(
            device), batch['length'].to(device), batch['target'].to(device)
        y_pred = pa_riskranker(cat_feat, conti_feat, length).squeeze()
        try:
            dev_ranking_score.extend(y_pred.detach().cpu().numpy())
        except:
            dev_ranking_score.append(y_pred.detach().cpu().numpy())
    dev_set['fst_step_scores'] = dev_ranking_score

    test_ranking_score = []
    test_set_loop = tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    test_set_loop.set_description("Computing ranking score for testing set")
    for batch in test_set_loop:
        cat_feat, conti_feat, length, target = batch['cat_feat'].to(device), batch['conti_feat'].to(
            device), batch['length'].to(device), batch['target'].to(device)
        y_pred = pa_riskranker(cat_feat, conti_feat, length).squeeze()
        try:
            test_ranking_score.extend(y_pred.detach().cpu().numpy())
        except:
            test_ranking_score.append(y_pred.detach().cpu().numpy())
    test_set['fst_step_scores'] = test_ranking_score

    # generate ranking labels for train and test set according to the ranking score
    train_set['ranking_label'] = train_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    dev_set['ranking_label'] = dev_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    test_set['ranking_label'] = test_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    train_set['ranking_label'] = train_set['ranking_label'].astype(int)
    dev_set['ranking_label'] = dev_set['ranking_label'].astype(int)
    test_set['ranking_label'] = test_set['ranking_label'].astype(int)

    # shuffle the data
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    dev_set = dev_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    # save the train and test set
    train_set.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    dev_set.to_csv(os.path.join(output_path, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(output_path, 'test.csv'), index=False)

    # compute the average ndcg score for each qid for train and test set
    # use proper evaluation metric ndcg@k depending on the group size
    print("Computing NDCG score...")

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
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_parser.add_argument("--optimizer", type=str, default="adam")
    train_parser.add_argument("--group_size", type=int, choices=[10, 20, 30, 50, 100, 200, 500, 1000], default=100)
    train_parser.add_argument("--dataset", type=str, required=True)
    train_parser.add_argument("--tf_dim_feedforward", type=int, default=128)
    train_parser.add_argument("--tf_nhead", type=int, default=4)
    train_parser.add_argument("--tf_num_layers", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.25)
    train_parser.add_argument("--over_sample", type=bool, default=False)
    train_parser.add_argument("--strategy", type=str, choices=['ordinal', 'binary'], default='binary')
    train_parser.add_argument("--pnl", action="store_true")
    train_parser.add_argument("--loss_fn", type=str, default="graph")
    train_parser.add_argument("--fold", type=int, default=1)
    train_parser.add_argument("--ft_embedding_checkpoint", type=str, default=None)
    train_parser.add_argument("--freeze_ft_embedder", action="store_true", default=False,
                             help="Whether to freeze the ft embedder during training")

    train_parser.set_defaults(func=train_pa_riskranker)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_group_size', type=str, choices=["20", "30", "50", "100", "200", "500", "1000"],
                             required=True, default=None)
    test_parser.add_argument('--test_group_size', type=str, choices=["20", "30", "50", "100", "200", "500", "1000"],
                             required=True, default=None)
    test_parser.add_argument("--dataset", type=str, required=True)
    test_parser.add_argument("--fold", type=int, default=1)
    test_parser.add_argument('--strategy', type=str, choices=['ordinal', 'binary'], required=True, default=None)
    test_parser.add_argument("--loss_fn", type=str, default="graph")
    test_parser.set_defaults(func=test_pa_riskranker)
    args = parser.parse_args()
    args.func(args)