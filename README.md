# Predicting-FDA-Drug-Approval-Timelines
Predicting FDA Drug Approval Timelines Using Clinical Trial Metadata
🎯 Objective:
Build a machine learning model to predict the time it will take for a drug to get FDA approval based on metadata from clinical trials (e.g., trial phase, sponsor type, number of participants, etc.).

📦 Project Structure
fda_approval_prediction/
├── data/
│   └── clinical_trials.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── requirements.txt
└── README.md

📊 Data Sources (Free & Public)
ClinicalTrials.gov API / dataset dump

Get metadata like: start/end dates, phase, sponsor type, enrollment numbers.

FDA Drug Approval Database:

FDA’s Drugs@FDA database

You can combine these datasets by matching drugs by name or identifiers (e.g., NCT IDs or sponsor names).

🛠 Features to Use

| Feature                   | Description             |
| ------------------------- | ----------------------- |
| `trial_phase`             | Phase I, II, III, or IV |
| `enrollment`              | Number of participants  |
| `sponsor_type`            | Industry, NIH, Academic |
| `start_date` / `end_date` | Trial timeline          |
| `intervention_type`       | Drug, Biological, etc.  |
| `condition`               | Disease being studied   |
| `is_fda_regulated`        | Boolean                 |

Target: Time (in months or days) from trial start to FDA approval.

🧠 Models to Try
Linear Regression (baseline)

Random Forest Regressor

XGBoost

Gradient Boosted Trees

📈 Evaluation Metrics
MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

📘 Sample Code Snippet

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

df = pd.read_csv('data/clinical_trials.csv')

# Basic preprocessing
df['approval_timeline_days'] = (pd.to_datetime(df['approval_date']) - pd.to_datetime(df['start_date'])).dt.days
df = df.dropna(subset=['approval_timeline_days'])

# Encode categorical variables
df = pd.get_dummies(df, columns=['phase', 'sponsor_type'])

X = df.drop(['approval_timeline_days', 'drug_name'], axis=1)
y = df['approval_timeline_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

📌 Project Goals
 Collect and clean clinical trial data

 Engineer meaningful features

 Predict FDA approval time

 Build an explainable model (e.g., SHAP values)


 🔍 Stretch Goals
Use NLP to analyze descriptions of trials/interventions

Build a dashboard using Streamlit or Dash

Create a risk-score model to flag drugs with long approval timelines

✅ Sample Project Code: Predicting FDA Drug Approval Timelines

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Sample clinical trial metadata
data = {
    'trial_id': ['NCT001', 'NCT002', 'NCT003', 'NCT004', 'NCT005'],
    'start_date': ['2015-01-01', '2016-05-15', '2017-03-20', '2018-07-30', '2019-10-10'],
    'approval_date': ['2017-06-01', '2019-01-10', '2018-11-25', '2021-04-01', '2022-09-15'],
    'phase': ['Phase 3', 'Phase 2', 'Phase 3', 'Phase 1', 'Phase 2'],
    'sponsor_type': ['Industry', 'NIH', 'Industry', 'Academic', 'Industry'],
    'enrollment': [300, 150, 400, 100, 250]
}

# Create DataFrame
df = pd.DataFrame(data)
df['start_date'] = pd.to_datetime(df['start_date'])
df['approval_date'] = pd.to_datetime(df['approval_date'])

# Create target variable: number of days to approval
df['approval_timeline_days'] = (df['approval_date'] - df['start_date']).dt.days

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['phase', 'sponsor_type'], drop_first=True)

# Features and target
X = df_encoded.drop(['trial_id', 'start_date', 'approval_date', 'approval_timeline_days'], axis=1)
y = df_encoded['approval_timeline_days']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)




