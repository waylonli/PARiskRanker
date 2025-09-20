# 🚀 PARiskRanker: Learn to Rank Risky Investors  

[![ACM](https://img.shields.io/static/v1?label=ACM%20TOIS&message=10.1145/3768623&color=blue&logo=acm)](https://dl.acm.org/doi/10.1145/3768623) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Stars](https://img.shields.io/github/stars/waylonli/PARiskRanker?style=social)]()

🔥 Official implementation of our ACM TOIS 2025 paper:  
**“Learn to Rank Risky Investors: A Case Study of Predicting Retail Traders’ Behaviour and Profitability”**  
by [Weixian Waylon Li](https://orcid.org/0000-0002-4196-9462) and [Tiejun Ma](https://orcid.org/0000-0001-5545-6978).  

📄 Paper DOI: [10.1145/3768623](https://dl.acm.org/doi/10.1145/3768623)


## ✨ Highlights

- **Profit-Aware Risk Ranker (PA-RiskRanker)** reframes risky investor detection as a *ranking* problem rather than classification.  
- Introduces **PA-BCE loss** to integrate Profit & Loss (P&L) into LETOR training.  
- **Self-Cross-Trader Attention** captures both intra-trader and inter-trader dependencies.    


## 🛠️ Setup

```bash
conda create -n pariskranker python=3.10
conda activate pariskranker
pip install -r requirements.txt
````


## 📂 Data

1. Download preprocessed data [here](https://drive.google.com/file/d/11jE6cCo9bdXE6pl-BkrLkHfzM1Bt1PLL/view?usp=sharing).
2. Unzip into the project root. Expected structure:

```
├── data
│   ├── creditcard
│   │   ├── fold1 / fold2 / fold3
│   │   └── creditcard.csv
│   ├── jobprofit
│   │   ├── fold1 / fold2 / fold3
│   │   └── job_profitability.csv
├── evaluation
│   └── metrics.py
...
```


## 📦 Pre-trained Model (Optional)

👉 \[TODO: Add link] – Place it in the project root as with the `data` folder.


## 🎯 Reproducing Results

Run any of the following to benchmark:

```bash
# PARiskRanker
python run_pariskranker.py test \
  --model_group_size 100 --test_group_size 100 \
  --dataset creditcard --fold 1 \
  --strategy binary --loss_fn graph
```

We also provide scripts for **Rankformer**, **LambdaMART**, **SOUR**, and baselines (classification & anomaly detection).
See README sections for full commands.

📊 Final evaluation notebooks: [`notebook/eval.ipynb`](notebook/eval.ipynb)


## 🏋️ Train from Scratch

```bash
FOLD=1  # choose from 1,2,3
python run_pariskranker.py train \
  --epochs 200 --batch_size 128 \
  --group_size 100 --dataset creditcard \
  --fold $FOLD --strategy binary --pnl 1 \
  --loss_fn graph
```

Hyperparameters can be customised via `--help`.


## 📈 Citation

If you use this code, please cite:

```bibtex
@article{10.1145/3768623,
author = {Li, Weixian Waylon and Ma, Tiejun},
title = {Learn to Rank Risky Investors: A Case Study of Predicting Retail Traders’ Behaviour and Profitability},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3768623},
doi = {10.1145/3768623},
journal = {ACM Trans. Inf. Syst.},
month = sep,
keywords = {learning to rank, domain-specific application, individual behaviour modelling, risk assessment}
}
```


## 🙌 Acknowledgements

This work was conducted at the **Artificial Intelligence Applications Institute, School of Informatics, University of Edinburgh**.
