import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf

# 1. Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.reset_index(inplace=True)

# 2. Preprocessing
scaler = MinMaxScaler(feature_range=(0,1))
data['Close_scaled'] = scaler.fit_transform(np.array(data['Close']).reshape(-1,1))

# Feature: use last N days as input
N = 5
X, y = [], []
for i in range(N, len(data)):
    X.append(data['Close_scaled'].values[i-N:i])  # last N days
    y.append(data['Close_scaled'].values[i])      # next day
X, y = np.array(X), np.array(y)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# 7. Plot results
plt.figure(figsize=(10,6))
plt.plot(range(len(y_test)), y_test, label="Actual")
plt.plot(range(len(y_pred)), y_pred, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction (Random Forest)")
plt.xlabel("Days")
plt.ylabel("Scaled Close Price")
plt.show()
