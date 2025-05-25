import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import data_loader, data_cleaning

# Load data
file_path = "C:/Users/siddh/Downloads/retail_sales_dataset.csv"
df = data_loader.load_data(file_path)
print("Initial Data:")
print(df.head())

# Clean data
df = data_cleaning.clean(df)

# Basic EDA
print("\nData Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Visualize sales trends
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df_grouped = df.groupby('date')['monthly_sales'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_grouped, x='date', y='monthly_sales')
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Monthly Sales")
plt.tight_layout()
plt.savefig("outputs/plots/monthly_sales_trend.png")
plt.show()
