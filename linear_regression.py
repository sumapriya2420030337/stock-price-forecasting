import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

# 1. Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.reset_index(inplace=True)   # keep Date as a column
print("Dataset head:\n", data.head())

# 2. Preprocessing
scaler = MinMaxScaler(feature_range=(0,1))
data['Close_scaled'] = scaler.fit_transform(np.array(data['Close']).reshape(-1,1))

# 3. Train-test split
X = np.array(range(len(data))).reshape(-1,1)
y = data['Close_scaled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# 7. Plot results
plt.figure(figsize=(10,6))
plt.plot(data['Date'][-len(y_test):], y_test, label="Actual")
plt.plot(data['Date'][-len(y_test):], y_pred, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction (Linear Regression)")
plt.xlabel("Date")
plt.ylabel("Scaled Close Price")
plt.show()
# 8. Future prediction
future_days = 30        
future_X = np.array(range(len(data), len(data) + future_days)).reshape(-1,1)
future_pred = model.predict(future_X)
future_pred_rescaled = scaler.inverse_transform(future_pred)
print("Future predictions for next 30 days:\n", future_pred_rescaled)

# Plot future predictions
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Close'], label="Historical Close Price")
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
plt.plot(future_dates, future_pred_rescaled, label="Future Predictions", linestyle='----')
plt.legend()        
plt.title("Future Stock Price Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()
# 9. Save model and scaler
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved.")
# 10. Load model and scaler (example)
loaded_model = joblib.load('linear_regression_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
print("Model and scaler loaded.")
