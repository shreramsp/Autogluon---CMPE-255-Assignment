# AutoGluon — CMPE 255 Data Mining Assignment

This repo contains **5 Colab notebooks** and **walkthrough videos** demonstrating AutoGluon for tabular ML, multimodal tabular, and automatic feature engineering. Each notebook runs end-to-end in Google Colab.

---

## Notebooks (with datasets)

1. **01_ieee_fraud.ipynb — IEEE-CIS Fraud Detection (binary classification, ROC-AUC)**
   - Dataset (Kaggle): https://www.kaggle.com/competitions/ieee-fraud-detection/data  
   - Files used: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`, `sample_submission.csv`.

2. **02_california_housing.ipynb — California Housing (regression, RMSE + submission)**
   - Dataset (Kaggle): https://www.kaggle.com/competitions/californiahousing/data  
   - Files used: `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`.

3. **03_tabular_quick_start.ipynb — Tabular Quick Start (multi-class classification)**
   - Dataset (public URL, loaded on the fly):  
     `https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/`  
     (uses `train.csv` / `test.csv` from that path)

4. **04_tabular_multimodal.ipynb — Multimodal Tabular (text + tabular, multiclass)**
   - Dataset (Kaggle, CSV-only): https://www.kaggle.com/competitions/petfinder-adoption-prediction/data  
   - Files used: `train.csv`, `test.csv`, `sample_submission.csv`  
   - *Note:* Images are **not** used in this repo to keep the run lightweight; the notebook enables AutoMM so text (`Description`) is modeled alongside tabular features.

5. **05_feature_engineering.ipynb — Automatic Feature Engineering (regression demo)**
   - Dataset: **synthetic mixed-type data generated in the notebook** (floats, ints, datetime, categorical, short text). No external download required.

---

## How to run (Colab)

1. Open the notebook in Colab.  
2. **Runtime → Change runtime type**  
   - CPU is fine for 01/02/03/05.  
   - **T4 GPU recommended** for 04 (multimodal tabular with text).  
3. Run the first cell to **mount Google Drive** (artifacts are saved to Drive paths used in each notebook).  
4. Run the `pip install autogluon` cell.  
5. **Run all**. Each notebook prints a leaderboard and (where applicable) writes a submission CSV.

---

## Artifacts

Small run artifacts are checked in under `artifacts/`:
