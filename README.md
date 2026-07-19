# Credit Card Default Prediction - JSS x AIHI Internship Work

This repository contains the data analysis, feature engineering, and machine learning modelling work for predicting credit card default. The project is structured to process raw credit card client data, perform Exploratory Data Analysis (EDA), engineer meaningful behavioral features, and train optimized gradient boosting models.

---

## 📁 Repository Structure

```text
├── Collab Notebooks/
│   ├── V1 Credit_Card_Default_Prediction.ipynb       # Initial V1 modeling and exploration
│   └── Optimized Code/
│       ├── Raw_Data (1).ipynb                        # Step 1: Data cleaning, EDA, and initial processing
│       └── Final_Modelling (3).ipynb                 # Step 2: Feature engineering, tuning, and modeling
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv                       # Original dataset from UCI Machine Learning Repository
│   └── processed/
│       └── credit_card_default_clean.csv             # Cleaned dataset ready for modeling
├── outputs/
│   ├── figures/                                      # Saved charts, correlation heatmaps, SHAP plots, etc.
│   ├── models/                                       # Trained model files
│   └── tables/                                       # CSV tables of comparison metrics and summaries
├── requirements.txt                                  # List of local Python dependencies
└── README.md                                         # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/anshulkanswal6-dev/JSS-x-AIHI-Internship-Work.git
   cd JSS-x-AIHI-Internship-Work
   ```

2. **Install Local Dependencies:**
   Ensure you have Python installed. You can install all necessary packages (such as `scikit-learn`, `catboost`, `optuna`, `shap`, etc.) using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Notebooks:**
   You can run the notebooks locally using VS Code, Jupyter Lab, or re-upload them to Google Colab. The notebooks are configured to automatically detect the environment (Colab vs. Local) and adjust file paths dynamically.

---

## 📈 Project Workflow

### Phase 1: Data Cleaning & EDA (`Raw_Data (1).ipynb`)
* Explored demographics, credit limits, billing history, and payment status.
* Handled outliers and duplicate values.
* RENAMED targets and removed irrelevant identifiers (`ID` column).
* Conducted correlation analyses and saved visualization plots.

### Phase 2: Feature Engineering & Advanced Modeling (`Final_Modelling (3).ipynb`)
* Engineered **10 new behavioral features** to capture payment patterns over the 6-month history:
  1. `AVG_PAY_STATUS` – Average monthly repayment delay.
  2. `MAX_PAY_STATUS` – Maximum payment delay code.
  3. `DELAYED_MONTHS` – Count of delayed payment months.
  4. `SEVERE_DELAY_MONTHS` – Count of severely delayed payment months ($\ge 2$ months).
  5. `AVG_BILL_AMT` – Average billing amount.
  6. `AVG_PAY_AMT` – Average monthly payment.
  7. `TOTAL_BILL` – Total billed amount over 6 months.
  8. `TOTAL_PAYMENT` – Total paid amount over 6 months.
  9. `CREDIT_UTILIZATION` – Average bill relative to credit limit.
  10. `PAYMENT_RATIO` – Average payment relative to average billing.
* Evaluated baseline models (Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost).
* Performed hyperparameter tuning using **Optuna** for the best-performing model (CatBoost).
* Provided global and local model explainability using **SHAP (Shapley Additive exPlanations)**.

---

## 📊 Key Results

* The **CatBoost Classifier** tuned with **Optuna** achieved the highest performance.
* Feature importance and SHAP analysis showed that repayment status (`PAY_0`), `CREDIT_UTILIZATION`, and credit limits (`LIMIT_BAL`) are key predictors of default risk.
