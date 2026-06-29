# Customer-Churn-Prediction-System

## Overview

Customer churn refers to customers discontinuing a company's service. This project predicts whether a customer is likely to churn based on demographic information, account details, and service usage.

The system helps businesses identify at-risk customers and take proactive retention measures.

---

## Features

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Encoding and Scaling
- Machine Learning Models
- Churn Prediction
- Model Evaluation
- Feature Importance Analysis
- Interactive Streamlit Dashboard
- Visualizations and Insights

---

## Dataset

IBM Telco Customer Churn Dataset

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Target Variable:

- Churn
    - Yes
    - No

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- XGBoost
- Joblib
- Streamlit

---

## Project Structure

Customer-Churn-Prediction/

│

├── data/

│     └── Telco-Customer-Churn.csv

│

├── notebooks/

│     ├── EDA.ipynb

│     └── Model_Training.ipynb

│

├── models/

│     └── churn_model.pkl

│

├── app.py

├── requirements.txt

├── README.md

└── screenshots/

---

## Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Saving
8. Streamlit Deployment

---

## Machine Learning Models

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Input Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

---

## Output

Churn Prediction:

- Likely to Churn
- Not Likely to Churn

Probability Score:

Example:

Churn Probability = 87%

---

## Installation

Clone the repository

```bash
git clone https://github.com/username/Customer-Churn-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit app

```bash
streamlit run app.py
```

---

## Future Improvements

- Hyperparameter Tuning
- SHAP Explainability
- Customer Segmentation using K-Means
- Customer Lifetime Value Prediction
- Email Alert System
- Deployment on Streamlit Community Cloud

---

## Applications

- Telecom Companies
- Banks
- Insurance Companies
- Subscription Services
- E-commerce Platforms

---

## Author

Arghadeep Ghosh
Ankita Dey(AIML-A6/May-9814)

B.Tech CSE, KIIT University
