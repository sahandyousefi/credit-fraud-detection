# Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent credit card transactions in highly imbalanced datasets.

---

## Project Overview

This project addresses the challenge of identifying fraudulent transactions in a dataset where fraud represents only 0.0003% of all transactions. Multiple machine learning algorithms were tested and compared, with special emphasis on handling extreme class imbalance.

**Dataset Statistics:**
- Total Transactions: 6,362,620
- Fraudulent Cases: 16
- Fraud Rate: 0.0003%
- Transaction Types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN

**Final Results:**
- 100% fraud detection rate
- Zero false positives
- Perfect precision and recall on test set

---

## Project Structure
```
CREDIT-FRAUD-DETECTION/
│
├── data/
│   ├── AIML Dataset.csv                    (Download separately)
│   └── processed/                          (Preprocessed datasets)
│
├── notebooks/
│   ├── 01_EDA.ipynb                       (Exploratory Data Analysis)
│   ├── 02_Feature_Engineering.ipynb       (Feature creation and preprocessing)
│   └── 03_Model_Building.ipynb            (Model training and evaluation)
│
├── models/
│   ├── best_fraud_detection_model.pkl     (Trained model)
│   └── model_comparison.csv               (Performance metrics)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

**Source:** [Fraud Detection AI & ML Model and Analysis - Kaggle](https://www.kaggle.com/datasets/nasaamit007/fraud-detection-ai-and-ml-model-and-analysis)

**How to Download:**
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/nasaamit007/fraud-detection-ai-and-ml-model-and-analysis)
2. Click the "Download" button (requires free Kaggle account)
3. Extract the downloaded ZIP file
4. Place `AIML Dataset.csv` in the `data/` folder of this project

**Dataset Details:**
- File Size: Approximately 470 MB
- Format: CSV
- Rows: 6,362,620 transactions
- Columns: 11 features

**Features:**
- step: Time step of the transaction
- type: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- amount: Transaction amount
- nameOrig: Customer initiating the transaction
- oldbalanceOrg: Initial balance before transaction
- newbalanceOrig: New balance after transaction
- nameDest: Recipient of the transaction
- oldbalanceDest: Initial recipient balance before transaction
- newbalanceDest: New recipient balance after transaction
- isFraud: Target variable (1 = fraud, 0 = legitimate)
- isFlaggedFraud: System-flagged fraud indicator

---

## Feature Engineering

I created 9 additional features to improve model performance:

**Balance-Based Features:**
1. balanceChangeOrig - Change in originator's balance
2. balanceChangeDest - Change in destination's balance
3. amountToOldBalanceRatio - Ratio of transaction amount to original balance
4. originEmptied - Binary indicator if origin account was emptied
5. destIsNew - Binary indicator if destination account is new

**Transaction Pattern Features:**
6. type_encoded - Numerical encoding of transaction type
7. isHighRiskType - Binary flag for high-risk transaction types (TRANSFER, CASH_OUT)
8. hour - Extracted hour from step variable
9. day - Extracted day number from step variable

**Feature Importance:**
The three most important features for fraud detection were:
1. balanceChangeOrig (26.25%)
2. oldbalanceOrg (16.76%)
3. amountToOldBalanceRatio (16.54%)

---

## Models Tested

I evaluated seven different machine learning approaches:

1. Logistic Regression (with balanced class weights)
2. Random Forest (default settings)
3. Random Forest (with balanced class weights)
4. Random Forest (trained on SMOTE-resampled data)
5. XGBoost (default settings)
6. XGBoost (with scale_pos_weight parameter)
7. XGBoost (trained on SMOTE-resampled data)

---

## Results

**Model Performance Comparison:**

| Model | Precision | Recall | F1-Score | False Positives |
|-------|-----------|--------|----------|-----------------|
| Random Forest (Balanced) | 100.00% | 100.00% | 1.0000 | 0 |
| XGBoost (Default) | 100.00% | 100.00% | 1.0000 | 0 |
| Random Forest (SMOTE) | 100.00% | 100.00% | 1.0000 | 0 |
| XGBoost (Balanced) | 99.67% | 100.00% | 0.9983 | 1 |
| XGBoost (SMOTE) | 99.67% | 100.00% | 0.9983 | 1 |
| Random Forest (Default) | 100.00% | 66.67% | 0.8000 | 0 |
| Logistic Regression | 2.97% | 100.00% | 0.0577 | 98 |

---

## Best Model

**Selected Model: Random Forest with Balanced Class Weights**

This model was chosen because it achieved:
- Perfect precision (100%) - no false alarms
- Perfect recall (100%) - caught all fraudulent transactions
- Simpler training pipeline compared to SMOTE approaches
- Faster training time with lower memory requirements
- Equivalent performance to more complex methods

**Test Set Performance:**
- Fraudulent transactions detected: 3 out of 3
- False positives: 0
- True negatives: 1,272,521
- False negatives: 0

---

## Key Findings

**1. Class Imbalance Handling is Critical**
- Models without imbalance handling missed 33% of frauds
- Balanced class weights improved recall from 67% to 100%
- Both SMOTE and class weighting produced excellent results

**2. Feature Engineering Had Significant Impact**
- Engineered features were among the most predictive
- Balance change features captured fraud patterns effectively
- Transaction amount alone was insufficient for detection

**3. Fraud Pattern Analysis**
- All fraudulent transactions were TRANSFER type
- All frauds involved exactly $10,000,000
- Fraudulent transfers typically emptied the origin account
- Clear separation between fraudulent and legitimate patterns

**4. Model Comparison Insights**
- Tree-based models outperformed linear models
- Default Random Forest struggled with extreme imbalance
- Logistic Regression produced 98 false positives despite high recall
- Simple class weighting was as effective as complex resampling

---

## Installation and Usage

**Requirements:**
- Python 3.8 or higher
- Free Kaggle account (for dataset download)

**Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/credit-fraud-detection.git
cd credit-fraud-detection
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Download the Dataset**
1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/nasaamit007/fraud-detection-ai-and-ml-model-and-analysis)
2. Click "Download" (sign in to Kaggle if prompted)
3. Extract the downloaded file
4. Move `AIML Dataset.csv` to the `data/` folder

**Step 4: Run the Analysis**

Start Jupyter Notebook:
```bash
jupyter notebook
```

Run the notebooks in this order:
1. `notebooks/01_EDA.ipynb` - Exploratory data analysis
2. `notebooks/02_Feature_Engineering.ipynb` - Feature creation and preprocessing
3. `notebooks/03_Model_Building.ipynb` - Model training and evaluation

**Using the Trained Model:**
```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('models/best_fraud_detection_model.pkl')

# Make predictions on new data
predictions = model.predict(new_transaction_data)
```

---

## Technical Details

**Handling Extreme Class Imbalance:**
- Applied balanced class weights to increase penalty for minority class misclassification
- Implemented SMOTE (Synthetic Minority Over-sampling Technique) with 1% sampling strategy
- Compared multiple imbalance handling approaches
- Evaluated using precision, recall, and F1-score rather than accuracy

**Data Preprocessing:**
- Label encoding for categorical transaction types
- StandardScaler for feature normalization
- Train-test split (80/20) with stratification to maintain fraud ratio
- Handled missing values and infinite values in engineered features

**Model Evaluation Strategy:**
- Primary focus on recall (minimizing missed frauds)
- Secondary focus on precision (minimizing false alarms)
- Confusion matrix analysis for detailed error understanding
- ROC-AUC curves for model discrimination comparison
- Business impact analysis of prediction errors

---

## Technologies Used

- **Python 3.8+** - Programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and tools
- **imbalanced-learn** - SMOTE and imbalance handling techniques
- **xgboost** - Gradient boosting framework
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **joblib** - Model serialization and persistence
- **jupyter** - Interactive development environment

---

## Limitations

- Small number of fraud cases in dataset (16 total, 3 in test set)
- All fraudulent transactions follow the same pattern (high-value transfers)
- Model not tested on diverse fraud types or real-world scenarios
- No temporal analysis or time-series patterns explored
- Results may not generalize to other fraud detection contexts

---

## Future Improvements

- Deploy model as REST API using Flask or FastAPI
- Add model explainability using SHAP values or LIME
- Implement real-time fraud detection pipeline
- Conduct hyperparameter tuning using grid search or Optuna
- Add anomaly detection models for unknown fraud patterns
- Implement time-based cross-validation
- Create monitoring dashboard for production deployment
- Test ensemble methods combining multiple models
- Evaluate on additional fraud detection datasets

---

## License

This project is part of my data science portfolio.

---

## Acknowledgments

- Dataset provided by [nasaamit007](https://www.kaggle.com/datasets/nasaamit007/fraud-detection-ai-and-ml-model-and-analysis) on Kaggle
- Project completed as part of machine learning portfolio development
- Inspired by real-world fraud detection challenges in financial systems

---

## Contact

**Sahand Yousefi**

- GitHub: [@sahandyousefi](https://github.com/sahandyousefi)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/sahand-yousefi/)

For questions or feedback about this project, feel free to reach out or open an issue in the repository.