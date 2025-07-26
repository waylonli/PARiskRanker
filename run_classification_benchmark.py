import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from pytorch_tabular.utils import get_class_weighted_cross_entropy

from evaluation.metrics import evaluate

def train_and_test_model(model, dataset):
    if dataset == 'creditcard':
        variables = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]
        variables_cat = []
        train_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/train.csv')
        val_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/val.csv')
        test_set = pd.read_csv(f'data/creditcard/fold{str(args.fold)}/100/test.csv')

    elif dataset == 'jobprofit':
        # bedrooms,bathrooms,sqft_living,floors,waterfront,view,condition
        variables_num = ['Jobs_Gross_Margin_Percentage', 'Labor_Pay', 'Labor_Burden', 'Material_Costs', 'PO_Costs', 'Equipment_Costs', 'Materials__Equip__POs_As_percent_of_Sales', 'Labor_Burden_as_percent_of_Sales', 'Labor_Pay_as_percent_of_Sales', 'Sold_Hours', 'Total_Hours_Worked', 'Total_Technician_Paid_Time', 'NonBillable_Hours', 'Jobs_Total_Costs', 'Jobs_Estimate_Sales_Subtotal', 'Jobs_Estimate_Sales_Installed', 'Materials__Equipment__PO_Costs']
        variables_cat = ['Is_Lead', 'Opportunity', 'Warranty', 'Recall', 'Converted', 'Estimates']
        variables = variables_num + variables_cat
        train_set = pd.read_csv(f'data/jobprofit/fold{str(args.fold)}/100/train.csv')
        val_set = pd.read_csv(f'data/jobprofit/fold{str(args.fold)}/100/val.csv')
        test_set = pd.read_csv(f'data/jobprofit/fold{str(args.fold)}/100/test.csv')

    else:
        raise Exception('Dataset not implemented')


    X_train = train_set[variables]
    y_train = train_set["Class"]
    X_val = val_set[variables]
    y_val = val_set["Class"]
    X_test = test_set[variables]
    y_test = test_set["Class"]

    if model == 'rf':
        model = RandomForestClassifier(verbose=True, n_jobs=-1, max_depth=7).fit(X_train, y_train)
    elif model == 'xgb':
        model = XGBClassifier(n_estimators=10000).fit(X_train, y_train)
    elif model == 'lgbm':
        model = LGBMClassifier(n_estimators=10000).fit(X_train, y_train)
    elif model == 'tabtransformer':
        run_tabtransformer(train_set, val_set, test_set, variables, dataset, cat_variables=variables_cat)
        return
    else:
        raise Exception('Model not implemented')

    # generate first step scores for train set
    try:
        train_set['fst_step_scores'] = model.predict_proba(X_train).iloc[:,1].values
    except:
        train_set['fst_step_scores'] = model.predict_proba(X_train)[:, 1]
    train_set['fst_step_pred'] = model.predict(X_train)

    # generate first step scores for val set
    try:
        val_set['fst_step_scores'] = model.predict_proba(X_val).iloc[:,1].values
    except:
        val_set['fst_step_scores'] = model.predict_proba(X_val)[:, 1]
    val_set['fst_step_pred'] = model.predict(X_val)

    # generate first step scores for test set
    try:
        test_set['fst_step_scores'] = model.predict_proba(X_test).iloc[:,1].values
    except:
        test_set['fst_step_scores'] = model.predict_proba(X_test)[:, 1]
    test_set['fst_step_pred'] = model.predict(X_test)

    model_path = os.path.join('storage', dataset, '{}_benchmark_fold{}'.format(args.model, str(args.fold)))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_set.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(model_path, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(model_path, 'test.csv'), index=False)

    return

def run_tabtransformer(train_set, val_set, test_set, variables, dataset, cat_variables=[]):
    num_cols = variables
    data_config = DataConfig(
        target=['Class'],
        # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
        continuous_cols=num_cols,
        # categorical_cols=cat_variables,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
        batch_size=128,
        min_epochs=100,
        max_epochs=500,
        check_val_every_n_epoch=10,
        accelerator="auto",  # can be 'cpu','gpu', 'tpu', or 'ipu'
        devices=1,
    )
    optimizer_config = OptimizerConfig()

    head_config = LinearHeadConfig(
        layers="",  # No additional layer in head, just a mapping layer to output_dim
        dropout=0.1,
        initialization="kaiming"
    ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

    model_config = TabTransformerConfig(
        task="classification",
        target_range=[(0, 1)],
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    weighted_loss = get_class_weighted_cross_entropy(train_set["Class"].values.ravel(), mu=0.1)

    model.fit(train=train_set, validation=val_set, loss=weighted_loss)

    # generate first step scores for train set
    pred_df = model.predict(train_set)

    y_proba_1 = pred_df['Class_1_probability'].values
    y_pred = pred_df['Class_prediction'].values
    train_set['fst_step_scores'] = y_proba_1
    train_set['fst_step_pred'] = y_pred

    # generate first step scores for val set
    pred_df = model.predict(val_set)
    y_proba_1 = pred_df['Class_1_probability'].values
    y_pred = pred_df['Class_prediction'].values
    val_set['fst_step_scores'] = y_proba_1
    val_set['fst_step_pred'] = y_pred

    # generate first step scores for test set
    pred_df = model.predict(test_set)
    y_proba_1 = pred_df['Class_1_probability'].values
    y_pred = pred_df['Class_prediction'].values
    test_set['fst_step_scores'] = y_proba_1
    test_set['fst_step_pred'] = y_pred

    model_path = os.path.join('storage', dataset, '{}_benchmark_fold{}'.format(args.model, args.fold))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # save model
    model.save_model(model_path)
    train_set.to_csv(os.path.join(model_path, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(model_path, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(model_path, 'test.csv'), index=False)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    print("===== Running {} benchmark =====".format(args.model))
    train_and_test_model(args.model, dataset=args.dataset)