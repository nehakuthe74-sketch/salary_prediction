# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# ==============================
# 2. LOAD DATA (FIXED URL)
# ==============================
url = "https://raw.githubusercontent.com/nehakuthe74-sketch/salary_prediction/main/Salary%20Data.csv"
df = pd.read_csv(url)

# ==============================
# 3. BASIC CLEANING
# ==============================
df.columns = df.columns.str.strip()

print("Columns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Convert to numeric safely
df = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values
df = df.dropna()

# ==============================
# 4. DEFINE FEATURES
# ==============================
X = df[['YearsExperience']]
y = df['Salary']

# ==============================
# 5. TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. LINEAR REGRESSION MODEL
# ==============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# Metrics
r2_lr = r2_score(y_test, y_pred_lr)

print("\n📈 Linear Regression R2:", r2_lr)

# ==============================
# 7. RANDOM FOREST (BEST MODEL)
# ==============================
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred_rf)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Performance:")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.2f}")

# ==============================
# 8. VISUALIZATION (CLEAN)
# ==============================
plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred_lr)
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# ==============================
# 9. SAVE MODEL
# ==============================
with open("salary_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\n✅ Model saved successfully!")

# ==============================
# 10. SAMPLE PREDICTION
# ==============================
sample = np.array([[5]])
prediction = rf_model.predict(sample)

print(f"\n💰 Predicted salary for 5 years experience: {prediction[0]:.2f}")
### Visual Comparison of Model Performance
"""

