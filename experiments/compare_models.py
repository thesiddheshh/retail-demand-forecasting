import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import data_loader, data_cleaning, feature_engineering, model_arima, model_lstm, forecast, metrics, visualize

# Load and clean data
df = data_loader.load_data("C:/Users/siddh/Downloads/retail_sales_dataset.csv")
df = data_cleaning.clean(df)

# Focus on Product A
df_product = df[df["product_name"] == "Product A"]
df_monthly = feature_engineering.aggregate_monthly_sales(df_product)

# Train-test split
train, test = df_monthly[:-12], df_monthly[-12:]

# --- ARIMA ---
arima_model = model_arima.train(train)
arima_preds = forecast.predict(arima_model, periods=12)
arima_rmse = metrics.compute_rmse(test["monthly_sales"], arima_preds)

# --- LSTM ---
X_train, y_train, scaler = model_lstm.prepare_data(train)
lstm_model = model_lstm.train(X_train, y_train)
lstm_preds = forecast.predict_lstm(lstm_model, train, scaler, n_future=12)
lstm_rmse = metrics.compute_rmse(test["monthly_sales"], lstm_preds)

# Print results
print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# Visual comparison
visualize.compare_forecasts(test["monthly_sales"], arima_preds, lstm_preds)
