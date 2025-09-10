# 📈 Stock Price Prediction using Machine Learning  

This project predicts stock prices using **Linear Regression** and **Random Forest Regression** models on real stock market data (fetched using `yfinance`).  

## 🔹 Features
- Fetches stock data directly from **Yahoo Finance**  
- Preprocessing: scaling, train-test split, normalization  
- Models: **Linear Regression** & **Random Forest Regression**  
- Evaluation using **RMSE (Root Mean Squared Error)**  
- Visualization of actual vs predicted stock prices  
- Saved trained models using `joblib` for reusability  

## 📊 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- yFinance  

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/your-username/stock-price-predictor.git

cd stock-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run Linear Regression version
python linear_regression_stock.py

# Run Random Forest version
python stock_pred.py
