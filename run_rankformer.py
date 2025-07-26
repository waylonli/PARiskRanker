from datetime import datetime

import numpy as np

import util
from ranking_model.rankformer import RankFormer
from ranking_model.loss import MSELoss, SoftmaxLoss, LambdaLoss
import argparse
import os
import pandas as pd
import json
import torch
from preprocess.dataloader import load_data
from tqdm import tqdm
from sklearn.metrics import ndcg_score

def train_rankformer(args):
    args.head_hidden_layers = [int(layer) for layer in
                               args.head_hidden_layers.replace('[', '').replace(']', '').split(',')]
    # define variables
    if args.dataset == 'creditcard':
        variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]


    elif args.dataset == "jobprofit":
        conti_variables = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs',
                           'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales',
                           'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours',
                           'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs',
                           'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed',
                           'Materials__Equipment__PO_Costs']
        categ_variables = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
        variables = conti_variables + categ_variables

    else:
        raise ValueError("Dataset not supported")

    batch_size_map = {20: 512, 50: 512, 100: 256, 200: 256, 500: 128, 1000: 32}
    args.batch_size = batch_size_map[args.group_size]

    # Load data to data loader
    # get the absolute path of the current file
    label_column = "Class"

    train_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'train.csv'))
    val_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', str(args.group_size), 'val.csv'))

    train_loader, input_dim, max_target = load_data(train_set, stage='train', variables=variables, batch_size=args.batch_size, label_column=label_column)
    dev_loader, _, _ = load_data(val_set, stage='dev', variables=variables, batch_size=args.batch_size, label_column=label_column)


    # init model
    rankformer = RankFormer(input_dim=len(variables),
                            tf_dim_feedforward=args.tf_dim_feedforward,
                            tf_nhead=args.tf_nhead,
                            tf_num_layers=args.tf_num_layers,
                            head_hidden_layers=args.head_hidden_layers,
                            dropout=args.dropout,
                            output_embedding_mode=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rankformer.to(device)

    # save the config
    if not os.path.exists("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold))):
        os.makedirs("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold)))
    config_dict = rankformer.export_config_dict()
    config_dict['group_size'] = args.group_size
    config_dict['strategy'] = args.strategy
    with open(os.path.join("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold)), 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     rankformer = torch.nn.DataParallel(rankformer)

    # init optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(rankformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(rankformer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer not supported")

    # init scheduler
    decay_fn = lambda t: (t - 10 + 1) ** -.005 if t >= 10 else 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_fn)

    # init loss function
    if args.loss == 'softmax':
        criterion = SoftmaxLoss()
    elif args.loss == 'mse':
        criterion = MSELoss()
    elif args.loss == 'lambdaloss':
        criterion = LambdaLoss()
    else:
        raise ValueError("Loss function not supported")

    # start training
    print("Start training...")
    best_dev_loss = float('inf')
    loop = tqdm(range(1, args.epochs+1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    since_last_improvement = 0

    for epoch in loop:
        log_loss = 0
        since_last_improvement += 1
        rankformer.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
            optimizer.zero_grad()
            y_pred = rankformer(feat, length)
            loss = criterion(y_pred, target) / args.group_size if args.strategy == 'ordinal' else criterion(y_pred, target)
            log_loss += loss.item()
            loss.backward()
            optimizer.step()

        loop.set_description(f"Epoch {epoch} | Train Loss {round(log_loss / len(train_loader), 4)}")
        # evaluate on dev set every 10 epochs and save the best model
        if epoch % 5 == 0:
            rankformer.eval()
            dev_loss = 0
            for batch in dev_loader:
                feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
                y_pred = rankformer(feat, length)
                loss = criterion(y_pred, target) / args.group_size if args.strategy == 'ordinal' else criterion(y_pred, target)
                dev_loss += loss.item()
            dev_loss /= len(dev_loader)

            loop.set_description(f"Epoch {epoch} | Train Loss {round(log_loss / len(train_loader), 4)} | Dev Loss {round(dev_loss, 4)}")
            if dev_loss < best_dev_loss:
                since_last_improvement = 0
                print(f"Saving best model at epoch {epoch}")
                best_dev_loss = dev_loss
                if os.path.exists(os.path.join("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold)), 'best_model.pt')):
                    os.remove(os.path.join("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold)), 'best_model.pt'))
                torch.save(rankformer.state_dict(), os.path.join("./storage/{}/rankformer_{}_{}_{}_fold{}".format(args.dataset, args.group_size, args.strategy, args.loss, str(args.fold)), 'best_model.pt'))
                # save the config file
        # early stopping if the dev loss does not decrease for 50 epochs
        if since_last_improvement >= 60:
            print(f"Early stopping at epoch {epoch}")
            break
        scheduler.step()

    return

def test_rankformer(args):
    if args.dataset == 'creditcard':
        variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
    elif args.dataset == "jobprofit":
        conti_variables = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs',
                           'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales',
                           'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours',
                           'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs',
                           'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed',
                           'Materials__Equipment__PO_Costs']
        categ_variables = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
        variables = conti_variables + categ_variables
    else:
        raise ValueError("Dataset not supported")

    # load the best model and config
    model_path = os.path.join('storage', args.dataset, 'rankformer_{}_{}_{}_fold{}'.format(args.model_group_size, args.strategy, args.loss, str(args.fold)))
    output_path = os.path.join(model_path, args.test_group_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    config['head_hidden_layers'] = [int(layer) for layer in config['head_hidden_layers']]
    rankformer = RankFormer(input_dim=config['input_dim'],
                            tf_dim_feedforward=config['tf_dim_feedforward'],
                            tf_nhead=config['tf_nhead'],
                            tf_num_layers=config['tf_num_layers'],
                            head_hidden_layers=config['head_hidden_layers'],
                            dropout=config['dropout'],
                            output_embedding_mode=False)
    rankformer.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pt')))
    rankformer.to(device)
    rankformer.eval()
    train_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', args.test_group_size, 'train.csv')).sort_values(by=['qid', 'Class'])
    dev_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', args.test_group_size, 'val.csv')).sort_values(by=['qid', 'Class'])
    test_set = pd.read_csv(os.path.join('data', args.dataset, f'fold{str(args.fold)}', args.test_group_size, 'test.csv')).sort_values(by=['qid', 'Class'])

    train_loader, _, _ = load_data(train_set, stage='test', variables=variables, label_column='Class', batch_size=1)
    dev_loader, _, _ = load_data(dev_set, stage='test', variables=variables, label_column='Class', batch_size=1)
    test_loader, _, _ = load_data(test_set, stage='test', variables=variables, label_column='Class', batch_size=1)

    train_ranking_score = []
    train_set_loop = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    train_set_loop.set_description("Computing ranking score for training set")
    for batch in train_set_loop:
        feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
        y_pred = rankformer(feat, length)
        train_ranking_score.extend(y_pred.detach().cpu().numpy())
    train_set['fst_step_scores'] = train_ranking_score

    dev_ranking_score = []
    dev_set_loop = tqdm(dev_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    dev_set_loop.set_description("Computing ranking score for dev set")
    for batch in dev_set_loop:
        feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
        y_pred = rankformer(feat, length)
        dev_ranking_score.extend(y_pred.detach().cpu().numpy())
    dev_set['fst_step_scores'] = dev_ranking_score

    test_ranking_score = []
    test_set_loop = tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    test_set_loop.set_description("Computing ranking score for testing set")
    for batch in test_set_loop:
        feat, length, target = batch['feat'].to(device), batch['length'].to(device), batch['target'].to(device)
        y_pred = rankformer(feat, length)
        test_ranking_score.extend(y_pred.detach().cpu().numpy())
    test_set['fst_step_scores'] = test_ranking_score

    # generate ranking labels for train and test set according to the ranking score
    train_set['ranking_label'] = train_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    dev_set['ranking_label'] = dev_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    test_set['ranking_label'] = test_set.groupby('qid')['fst_step_scores'].rank(method='dense', ascending=False)
    train_set['ranking_label'] = train_set['ranking_label'].astype(int)
    dev_set['ranking_label'] = dev_set['ranking_label'].astype(int)
    test_set['ranking_label'] = test_set['ranking_label'].astype(int)

    # shuffle the data
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    dev_set = dev_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    # save the train and test set
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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
    print("Test NDCG@3: {:.4f}".format(test_ndcg_3))
    print("Test NDCG@5: {:.4f}".format(test_ndcg_5))
    print("Test NDCG@10: {:.4f}".format(test_ndcg_10))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--dataset", type=str, required=True)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--optimizer", type=str, default="adam")
    train_parser.add_argument("--group_size", type=int, choices=[20, 50, 100, 200, 500, 1000], default=100)
    train_parser.add_argument("--tf_dim_feedforward", type=int, default=128)
    train_parser.add_argument("--tf_nhead", type=int, default=4)
    train_parser.add_argument("--tf_num_layers", type=int, default=4)
    train_parser.add_argument("--head_hidden_layers", type=str, default="[32]")
    train_parser.add_argument("--dropout", type=float, default=0.25)
    train_parser.add_argument("--over_sample", type=bool, default=False)
    train_parser.add_argument("--fold", type=int, default=1)
    train_parser.add_argument("--strategy", type=str, choices=['ordinal', 'binary'], default='binary')
    train_parser.add_argument("--loss", type=str, choices=['softmax', 'lambdaloss'], default='softmax')
    train_parser.set_defaults(func=train_rankformer)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--model_group_size', type=str, choices=["20", "30", "50", "100", "200", "500", "1000"], required=True, default=None)
    test_parser.add_argument("--dataset", type=str, default="creditcard")
    test_parser.add_argument('--test_group_size', type=str, choices=["20", "30", "50", "100", "200", "500", "1000"], required=True, default=None)
    test_parser.add_argument('--strategy', type=str, choices=['ordinal', 'binary'], required=True, default='binary')
    test_parser.add_argument("--fold", type=int, default=1)
    test_parser.add_argument("--loss", type=str, choices=['softmax', 'lambdaloss'], default='softmax')
    test_parser.set_defaults(func=test_rankformer)
    args = parser.parse_args()

    args.func(args)