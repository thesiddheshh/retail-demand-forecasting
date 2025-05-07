import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

forecast_steps = 12
df_data, df_full, forecasts = None, None, {}

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.drop_duplicates()
    df['product_name'] = df['product_name'].fillna('Unknown Product')
    df['product_cost'] = df['product_cost'].fillna(df['product_cost'].median())
    df['monthly_sales'] = df['monthly_sales'].fillna(0)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df_grouped = df.groupby('date').agg(monthly_sales=('monthly_sales', 'sum')).reset_index()
    df_grouped = df_grouped.sort_values('date').set_index('date')
    return df_grouped, df

def arima_forecast(data):
    model = SARIMAX(data['monthly_sales'], order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit(disp=False)
    forecast = result.forecast(steps=forecast_steps)
    return forecast, result

def regression_forecast(data, model):
    df = data.copy()
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['monthly_sales'].shift(1)
    df['lag_2'] = df['monthly_sales'].shift(2)
    df['lag_3'] = df['monthly_sales'].shift(3)
    df.dropna(inplace=True)

    features = ['month', 'year', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['monthly_sales']

    model.fit(X, y)
    last = df.iloc[-1]
    future = []

    for _ in range(forecast_steps):
        month = last.name.month + 1 if last.name.month < 12 else 1
        year = last.name.year + 1 if last.name.month == 12 else last.name.year
        input_features = [month, year, last['monthly_sales'], last['lag_1'], last['lag_2']]
        pred = model.predict([input_features])[0]
        future.append(pred)

        row = pd.Series({
            'monthly_sales': pred,
            'lag_1': last['monthly_sales'],
            'lag_2': last['lag_1'],
            'lag_3': last['lag_2']
        }, name=pd.Timestamp(year=year, month=month, day=1))

        df = pd.concat([df, pd.DataFrame([row])])
        last = df.iloc[-1]

    return future, model

def gbdt_forecast(data): return regression_forecast(data, GradientBoostingRegressor())
def rf_forecast(data): return regression_forecast(data, RandomForestRegressor())
def lr_forecast(data): return regression_forecast(data, LinearRegression())

def create_dashboard():
    global df_data, df_full, forecasts


    def select_file():
        path = filedialog.askopenfilename(title="Select Retail Sales CSV", filetypes=[("CSV Files", "*.csv")])
        if path:
            file_path_var.set(path)
            label_file_path.config(text=os.path.basename(path))

    def update_forecasts():
        global df_data, df_full, forecasts
        path = file_path_var.get()
        if not path:
            messagebox.showerror("Error", "No file selected.")
            return
        try:
            df_data, df_full = load_and_prepare_data(path)
            forecasts = {
                'ARIMA': arima_forecast(df_data)[0],
                'GBDT': gbdt_forecast(df_data)[0],
                'Random Forest': rf_forecast(df_data)[0],
                'Linear Regression': lr_forecast(df_data)[0]
            }
            messagebox.showinfo("Done", "Forecasts generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def render_plot(fig):
        for widget in frame_plot.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_graph(model_name):
        if model_name not in forecasts:
            messagebox.showerror("Error", f"{model_name} not trained yet.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_data.index, df_data['monthly_sales'], label='Historical')
        future_idx = pd.date_range(df_data.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
        ax.plot(future_idx, forecasts[model_name], label=f'{model_name} Forecast', linestyle='--')
        ax.set_title(f"{model_name} Forecast")
        ax.legend()
        render_plot(fig)

    def show_product_comparison():
        product_sales = df_full.groupby('product_name')['monthly_sales'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        product_sales.plot(kind='barh', ax=ax, color='teal')
        ax.set_title("Product-wise Total Sales")
        render_plot(fig)

    def show_sales_heatmap():
        df = df_full.copy()
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        heatmap_data = df.pivot_table(index='month', columns='year', values='monthly_sales', aggfunc='sum')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
        ax.set_title("Sales Heatmap")
        render_plot(fig)

    def show_yearly_boxplot():
        df_full['year'] = df_full['date'].dt.year
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='year', y='monthly_sales', data=df_full, ax=ax)
        ax.set_title("Yearly Sales Distribution")
        render_plot(fig)

    # GUI Layout
    root = tk.Tk()

    file_path_var = tk.StringVar()

    root.title("Retail Forecasting Dashboard")
    root.geometry("1400x900")
    root.configure(bg='#f3e8ff')

    font = ("Times New Roman", 12)
    button_style = {'font': font, 'bg': '#90ee90', 'activebackground': '#76d76d', 'bd': 0}

    tk.Label(root, text="Retail Forecasting System", font=("Times New Roman", 20, "bold"), bg='#f3e8ff').pack(pady=10)

    file_section = tk.Frame(root, bg='#f3e8ff')
    file_section.pack(pady=5)
    tk.Button(file_section, text="Select File", command=select_file, **button_style).pack()
    label_file_path = tk.Label(file_section, text="", font=font, bg='#f3e8ff')
    label_file_path.pack()

    tk.Button(root, text="Upload Dataset", command=update_forecasts, **button_style, width=20).pack(pady=10)

    model_buttons = tk.Frame(root, bg='#f3e8ff')
    model_buttons.pack(pady=5)
    for model in ['ARIMA', 'GBDT', 'Random Forest', 'Linear Regression']:
        tk.Button(model_buttons, text=f"{model} Forecast", command=lambda m=model: show_graph(m), **button_style, width=18).pack(side=tk.LEFT, padx=5)

    graph_buttons = tk.Frame(root, bg='#f3e8ff')
    graph_buttons.pack(pady=10)
    tk.Button(graph_buttons, text="Product Comparison", command=show_product_comparison, **button_style).pack(side=tk.LEFT, padx=5)
    tk.Button(graph_buttons, text="Sales Heatmap", command=show_sales_heatmap, **button_style).pack(side=tk.LEFT, padx=5)
    tk.Button(graph_buttons, text="Yearly Boxplot", command=show_yearly_boxplot, **button_style).pack(side=tk.LEFT, padx=5)

    frame_plot = tk.Frame(root, bg='white')
    frame_plot.pack(pady=20, fill='both', expand=True)

    root.mainloop()

if __name__ == "__main__":
    create_dashboard()
