import sys
import os
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import data_loader, data_cleaning, feature_engineering, model_arima, forecast, metrics, visualize

# Load and clean data
df = data_loader.load_data("C:/Users/siddh/Downloads/retail_sales_dataset.csv")
df = data_cleaning.clean(df)

# Select product and aggregate
df_product = df[df["product_name"] == "Product A"]
df_monthly = feature_engineering.aggregate_monthly_sales(df_product)

# Train-test split
train, test = df_monthly[:-12], df_monthly[-12:]

# Train ARIMA model
arima_model = model_arima.train(train)
arima_preds = forecast.predict(arima_model, periods=12)

# Evaluate
rmse = metrics.compute_rmse(test["monthly_sales"], arima_preds)
print(f"ARIMA RMSE: {rmse:.2f}")

# Plot
visualize.plot_forecast(train, test, arima_preds, title="ARIMA Forecast")
