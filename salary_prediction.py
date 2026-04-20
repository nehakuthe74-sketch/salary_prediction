# -*- coding: utf-8 -*-
# Salary Prediction Project (Fixed Version)

import os
import pandas as pd
import numpy as np

# ==============================
# 1. LOAD DATA (FIXED)
# ==============================

file_path = "Salary Data.csv"  # Make sure file is in same folder

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        "Make sure 'Salary Data.csv' is in the same directory as this script."
    )

df = pd.read_csv(file_path)

print("Dataset loaded successfully!\n")
print(df.head())

# ==============================
# 2. HANDLE MISSING VALUES
# ==============================

# Numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# Categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nNull values after cleaning:\n", df.isnull().sum())

# ==============================
# 3. ENCODE CATEGORICAL DATA
# ==============================

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoded Data:\n", df.head())

# ==============================
# 4. SPLIT DATA
# ==============================

X = df.drop('Salary', axis=1)
y = df['Salary']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ==============================
# 5. TRAIN MODELS
# ==============================

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

models = {}

# Linear Regression
models['Linear Regression'] = LinearRegression()

# KNN (with scaling FIX)
models['KNN'] = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

# SVM
models['SVM'] = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVR(kernel='linear'))
])

# Decision Tree
models['Decision Tree'] = DecisionTreeRegressor(random_state=42)

# Random Forest
models['Random Forest'] = RandomForestRegressor(random_state=42)

# ==============================
# 6. TRAIN & EVALUATE
# ==============================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = r2

    print(f"\n{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")

# ==============================
# 7. VISUALIZATION (FIXED)
# ==============================

import matplotlib.pyplot as plt

model_names = list(results.keys())
r2_scores = list(results.values())

plt.figure()
plt.barh(model_names, r2_scores)
plt.xlabel("R-squared")
plt.title("Model Comparison")
plt.grid(True)
plt.show()

# ==============================
# 8. SAVE BEST MODEL
# ==============================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model is: {best_model_name}")
print("Model saved as 'best_model.pkl'")

