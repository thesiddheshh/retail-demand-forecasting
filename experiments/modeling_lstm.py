import sys
import os
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import data_loader, data_cleaning, feature_engineering, model_lstm, forecast, metrics, visualize

# Load and clean
df = data_loader.load_data("C:/Users/siddh/Downloads/retail_sales_dataset.csv")
df = data_cleaning.clean(df)

# Select and aggregate
df_product = df[df["product_name"] == "Product A"]
df_monthly = feature_engineering.aggregate_monthly_sales(df_product)

# Train-test split
train, test = df_monthly[:-12], df_monthly[-12:]

# Prepare and train LSTM
X_train, y_train, scaler = model_lstm.prepare_data(train)
lstm_model = model_lstm.train(X_train, y_train)

# Predict
lstm_preds = forecast.predict_lstm(lstm_model, train, scaler, n_future=12)

# Evaluate
rmse = metrics.compute_rmse(test["monthly_sales"], lstm_preds)
print(f"LSTM RMSE: {rmse:.2f}")

# Plot
visualize.plot_forecast(train, test, lstm_preds, title="LSTM Forecast")
