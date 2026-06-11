# Customer Churn Prediction


---

## What This Project Does 

Banks and subscription businesses lose money every time a customer leaves. The goal here is to **predict which customers are likely to churn** (close their account / cancel their subscription) *before* they actually do — so the business can reach out proactively with retention offers.

This project builds three machine learning classifiers, compares them, and selects the best one. You also get a clean prediction interface for both single customers and batch CSV files.

---

## Dataset

**Source:** [Bank Customer Churn Prediction — Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

| Column           | Description                                   |
|------------------|-----------------------------------------------|
| CreditScore      | Customer's credit score (300–850)             |
| Geography        | Country: France / Germany / Spain             |
| Gender           | Male / Female                                 |
| Age              | Customer's age                                |
| Tenure           | Years with the bank (0–10)                    |
| Balance          | Account balance (€)                           |
| NumOfProducts    | Number of bank products held (1–4)            |
| HasCrCard        | Has credit card? (1 = Yes)                    |
| IsActiveMember   | Active in the last period? (1 = Yes)          |
| EstimatedSalary  | Estimated annual salary (€)                   |
| **Exited**       | **Target: 1 = Churned, 0 = Stayed**           |

**Size:** ~10,000 rows  |  **Churn rate:** ~20%

---

## Project Structure

```
customer_churn_prediction/
│
├── main.py                    ← Run the full pipeline from here
│
├── data/
│   └── Churn_Modelling.csv    ← Dataset (download from Kaggle link above)
│
├── src/
│   ├── eda.py                 ← Exploratory Data Analysis
│   ├── preprocessing.py       ← Cleaning, feature engineering, encoding, scaling
│   ├── train.py               ← Model training + evaluation + plots
│   └── predict.py             ← Inference for single customer or batch CSV
│
├── models/                    ← Saved model files (auto-generated)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── plots/                     ← All charts (auto-generated)
│   ├── 01_churn_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_categorical_churn_rates.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_age_balance_scatter.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   ├── 08_precision_recall.png
│   ├── 09_feature_importance.png
│   └── 10_model_comparison.png
│
├── reports/
│   └── batch_predictions.csv  ← Batch prediction output (auto-generated)
│
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download from Kaggle and place it at:
```
data/Churn_Modelling.csv
```

### 3. Run the full pipeline
```bash
python main.py
```

Or run individual steps:
```bash
python main.py --eda        # EDA only
python main.py --train      # Training + evaluation only
python main.py --predict    # Prediction demo (must train first)
```

---

## Models Used

| Model                | Why it was included                                             |
|---------------------|-----------------------------------------------------------------|
| Logistic Regression  | Interpretable baseline; fast; good for understanding coefficients |
| Random Forest        | Handles non-linear patterns; robust to outliers; feature importance |
| Gradient Boosting    | Usually strongest; builds trees sequentially to fix errors       |

All models are trained with `class_weight='balanced'` or equivalent to handle the ~80/20 class imbalance.

---

## Key Findings (typical results)

- **Age** is the strongest predictor — customers over 50 churn at significantly higher rates
- **IsActiveMember = 0** nearly doubles churn probability
- **Geography = Germany** shows ~2× the churn rate of France/Spain
- **NumOfProducts > 2** is associated with very high churn (counter-intuitive but real)
- Customers with **zero balance** churn more (likely dormant accounts)

---

## Predict for a New Customer

```python
from src.predict import predict_single

result = predict_single({
    "CreditScore":     600,
    "Geography":       "Germany",
    "Gender":          "Female",
    "Age":             42,
    "Tenure":          3,
    "Balance":         125000,
    "NumOfProducts":   1,
    "HasCrCard":       1,
    "IsActiveMember":  0,
    "EstimatedSalary": 101348,
})

print(result)
# {'churn_probability': 0.72, 'will_churn': True, 'risk_level': 'High'}
```

---

## Evaluation Metrics

We use **ROC-AUC** as the primary metric (not accuracy) because:
- The dataset is imbalanced (~20% churn)
- A model that predicts "no churn" for everyone gets 80% accuracy but is useless
- ROC-AUC measures the model's ability to *rank* churners above non-churners

We also track **Average Precision** (area under the Precision-Recall curve) for the same reason.

---

*Built as part of the CodSoft Machine Learning Internship — Task 3*
