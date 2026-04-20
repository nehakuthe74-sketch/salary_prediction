# -*- coding: utf-8 -*-
"""salary prediction"""

import pandas as pd

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/nehakuthe74-sketch/salary_prediction/main/Salary%20Data.csv")

# Clean column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.isnull().sum())

# ==============================
# HANDLE MISSING VALUES
# ==============================
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# ==============================
# LABEL ENCODING
# ==============================
from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ==============================
# REMOVE DUPLICATES
# ==============================
df = df.drop_duplicates()

# ==============================
# 🔥 FIXED HERE
# ==============================
# Use ONLY correct feature
X = df[['YearsExperience']]   # ✅ FIX
y = df['Salary']

# ==============================
# SPLIT DATA
# ==============================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# RANDOM FOREST MODEL
# ==============================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# ==============================
# SAFE PREDICTION
# ==============================
import numpy as np

# Must match training shape
sample = np.array([[5]])   # 5 years experience

pred = rf_model.predict(sample)

print("Prediction for 5 years:", pred[0])

sample = np.array([[5]])
pred = rf_model.predict(sample)

print("Prediction for 5 years:", pred[0])

