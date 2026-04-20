# -*- coding: utf-8 -*-
"""salary prediction"""

import pandas as pd

# ✅ FIXED: Use GitHub RAW link instead of /content path
df = pd.read_csv("https://raw.githubusercontent.com/nehakuthe74-sketch/salary_prediction/main/Salary%20Data.csv")

print(df.head())

print(df.isnull().sum())

# ==============================
# HANDLE MISSING VALUES
# ==============================

# Impute numerical columns with mean
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())   # ✅ fixed inplace issue

# Impute categorical columns with mode
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Null values after imputation:")
print(df.isnull().sum())

# ==============================
# LABEL ENCODING (SAFE)
# ==============================
from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))   # ✅ safer
    label_encoders[col] = le

print("Label Encoding applied:")
print(df.head())

# ==============================
# REMOVE DUPLICATES
# ==============================
df = df.drop_duplicates()

# ==============================
# SPLIT DATA
# ==============================
X = df.drop('Salary', axis=1)
y = df['Salary']

print("X:\n", X.head())
print("y:\n", y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ==============================
# LINEAR REGRESSION
# ==============================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print("Linear Regression:")
print(mae, mse, rmse, r2)

# ==============================
# KNN
# ==============================
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

print("KNN R2:", r2_score(y_test, y_pred_knn))

# ==============================
# SVM
# ==============================
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVR(kernel='linear'))
])

pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)

print("SVM R2:", r2_score(y_test, y_pred_svm))

# ==============================
# DECISION TREE
# ==============================
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("Decision Tree R2:", r2_score(y_test, y_pred_dt))

# ==============================
# RANDOM FOREST
# ==============================
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# ==============================
# VISUALIZATION
# ==============================
import matplotlib.pyplot as plt
import seaborn as sns

model_names = ['Linear', 'KNN', 'SVM', 'Decision Tree', 'Random Forest']
r2_scores = [
    r2,
    r2_score(y_test, y_pred_knn),
    r2_score(y_test, y_pred_svm),
    r2_score(y_test, y_pred_dt),
    r2_score(y_test, y_pred_rf)
]

performance_df = pd.DataFrame({
    'Model': model_names,
    'R2': r2_scores
})

plt.figure()
sns.barplot(x='R2', y='Model', data=performance_df)
plt.title("Model Comparison")
plt.show()

# ==============================
# SAVE MODEL
# ==============================
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model saved!")

# ==============================
# SAMPLE PREDICTION
# ==============================
import numpy as np

sample = np.array([[5]])
pred = rf_model.predict(sample)

print("Prediction for 5 years:", pred[0])

