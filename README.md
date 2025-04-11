# PARiskRanker
 
Support codes and data for our paper "Learn to Rank Risky Investors: A Case Study of Predicting
Retail Traders’ Behaviour and Profitability".

## 1. Environmental Setup

create a conda environment with the following command:
```bash
conda create -n pariskranker python=3.10
```
activate the environment:
```bash
conda activate pariskranker
```
install the required packages:
```bash
pip install -r requirements.txt
```

## 2. Data Preparation

Download the preprocessed data from this [link](TODO).
Unzip the downloaded data folder and place it at the project root directory. The folder structure should look like this:
```
├── data
│   ├── creditcard (3folds)
│   │   ├── fold1
│   │   ├── fold2
│   │   └── fold3
│   └── creditcard.csv
├── evaluation
│   └── metrics.py
├── ...
```

## 3. (Optional) Download the Pre-trained Model

Download the pre-trained model from this [link](TODO).
Unzip the downloaded model folder and place it at the project root directory, similar to what we did for the data folder.

## 4. Reproduce the Results
The results presented in the paper can be found in the `notebook/eval.ipynb` file.

If you want to reproduce the results yourself, run the following command:
```
# PARiskRanker
python run_pariskranker.py test \
 --model_group_size 100 \
 --test_group_size 100 \
 --dataset creditcard \
 --fold 1 \ # choose from 1, 2, 3
 --strategy binary \
 --loss_fn graph
 
# Rankformer
python run_rankformer.py test \
 --model_group_size 100 \
 --test_group_size 100 \
 --dataset creditcard \
 --fold 1 \ # choose from 1, 2, 3
 --strategy binary \
 --loss_fn graph # choose from softmax and lambdaloss
 
# LambdaMART
python ranking_model/lambdamart.py \
 --group_size 100 \
 --ntrees 10000 \
 --fold 1 \ # choose from 1, 2, 3
 --strategy binary

# LambdaMART with PA-BCE loss
python ranking_model/lambdamart_pa.py \
 --group_size 100 \
 --n_estimators 10000 \
 --fold 1 \ # choose from 1, 2, 3
 --strategy binary \
 --objective pabce
 
# SOUR
python ranking_model/sour/run_sour.py \
 --group_size 100 \
 --dataset creditcard \
 --fold 1 \ # choose from 1, 2, 3
 --strategy binary 
 
# Classification benchmarks
python run_classification_benchmark.py
 --model rf \ # choose from rf, xgb, lgbm, tabtransformer
 --fold 1  # choose from 1, 2, 3
 
# Anomaly detection benchmarks
python run_outlier_benchmark.py
 --model deepsad \ # choose from deepsad, deepisolationforest, feawad, slad
 --fold 1  # choose from 1, 2, 3
```

Then run the blocks in `notebook/eval.ipynb` again to see the model performance.

## 5. Train PARiskRanker from scratch

```bash
FOLD=1 # choose from 1, 2, 3

python run_pariskranker.py train \
 --epochs 200 \
 --batch_size 128 \
 --group_size 100 \
 --dataset creditcard \
 --fold $FOLD \
 --strategy binary \
 --pnl 1 \
 --loss_fn graph
```

Other tunable hyperparameters are listed in `run_pariskranker.py`. You can also use the `--help` flag to see all the available options.