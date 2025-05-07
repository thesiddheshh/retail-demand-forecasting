import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tkinter as tk
from tkinter import filedialog, messagebox, ttk  # ✅ FIX: include ttk here
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams.update({'axes.grid': True})

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

def arima_forecast(data, forecast_steps):
    model = SARIMAX(data['monthly_sales'], order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit(disp=False)
    forecast = result.forecast(steps=forecast_steps)
    return forecast, result

def regression_forecast(data, model, forecast_steps):
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
    predictions = []
    for _ in range(forecast_steps):
        month = last.name.month + 1 if last.name.month < 12 else 1
        year = last.name.year + 1 if last.name.month == 12 else last.name.year
        input_features = [month, year, last['monthly_sales'], last['lag_1'], last['lag_2']]
        pred = model.predict([input_features])[0]
        predictions.append(pred)

        row = pd.Series({
            'monthly_sales': pred,
            'lag_1': last['monthly_sales'],
            'lag_2': last['lag_1'],
            'lag_3': last['lag_2']
        }, name=pd.Timestamp(year=year, month=month, day=1))

        df = pd.concat([df, pd.DataFrame([row])])
        last = df.iloc[-1]
    return predictions, model

def create_dashboard():
    root = tk.Tk()
    root.title("Retail Forecasting Dashboard")
    root.geometry("1400x900")
    root.configure(bg='#f3e8ff')

    file_path_var = tk.StringVar()
    forecast_steps = tk.IntVar(value=12)
    forecasts = {}
    df_data = pd.DataFrame()
    df_full = pd.DataFrame()

    style = tk.ttk.Style()
    style.configure('TButton', font=('Times New Roman', 10, 'bold'), background='#90ee90', foreground='black')

    def select_file():
        file_path = filedialog.askopenfilename(title="Select Retail Sales CSV", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            file_path_var.set(file_path)

    def update_forecasts():
        nonlocal df_data, df_full, forecasts
        try:
            df_data, df_full = load_and_prepare_data(file_path_var.get())
            steps = forecast_steps.get()
            forecasts['ARIMA'], _ = arima_forecast(df_data, steps)
            forecasts['GBDT'], _ = regression_forecast(df_data, GradientBoostingRegressor(), steps)
            forecasts['Random Forest'], _ = regression_forecast(df_data, RandomForestRegressor(), steps)
            forecasts['Linear Regression'], _ = regression_forecast(df_data, LinearRegression(), steps)
            messagebox.showinfo("Success", "Forecasts updated.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def render_plot(fig):
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_graph(model_name):
        if model_name not in forecasts:
            messagebox.showerror("Error", f"Model {model_name} not trained.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_data.index, df_data['monthly_sales'], label='Historical')
        future_index = pd.date_range(df_data.index[-1], periods=forecast_steps.get()+1, freq='MS')[1:]
        ax.plot(future_index, forecasts[model_name], label=f"{model_name} Forecast", linestyle='--')
        ax.set_title(f"{model_name} Forecast")
        ax.legend()
        render_plot(fig)

    def show_model_performance():
        y_true = df_data['monthly_sales'][-forecast_steps.get():]
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = {}
        for model, preds in forecasts.items():
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mae = mean_absolute_error(y_true, preds)
            scores[model] = [rmse, mae]
        df_scores = pd.DataFrame(scores, index=["RMSE", "MAE"]).T
        df_scores.plot.bar(ax=ax)
        ax.set_title("Model Scores")
        render_plot(fig)

    def show_residual_plot():
        y_true = df_data['monthly_sales'][-forecast_steps.get():]
        fig, ax = plt.subplots()
        for model, preds in forecasts.items():
            residuals = y_true.values - np.array(preds[:forecast_steps.get()])
            ax.plot(residuals, label=model)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_title("Model Residuals")
        ax.legend()
        render_plot(fig)

    def show_product_comparison():
        sales_by_product = df_full.groupby('product_name')['monthly_sales'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        sales_by_product.plot(kind='barh', ax=ax, color='teal')
        ax.set_title("Product Sales Comparison")
        render_plot(fig)

    def show_sales_heatmap():
        df_temp = df_full.copy()
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['year'] = df_temp['date'].dt.year
        heatmap_data = df_temp.pivot_table(index='month', columns='year', values='monthly_sales', aggfunc='sum')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".0f", ax=ax)
        ax.set_title("Sales Heatmap")
        render_plot(fig)

    def show_yearly_boxplot():
        df_temp = df_full.copy()
        df_temp['year'] = df_temp['date'].dt.year
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='year', y='monthly_sales', data=df_temp, ax=ax)
        ax.set_title("Yearly Sales Distribution")
        render_plot(fig)

    # Title
    tk.Label(root, text="Retail Forecasting System", font=("Times New Roman", 20, "bold"), bg='#f3e8ff').pack(pady=10)

    # File selection
    file_frame = tk.Frame(root, bg='#f3e8ff')
    file_frame.pack(pady=5)

    tk.Button(file_frame, text="Select File", command=select_file, bg='#90ee90', font=("Times New Roman", 10, "bold")).pack()
    tk.Label(file_frame, textvariable=file_path_var, bg='#f3e8ff', font=("Times New Roman", 10)).pack()

    # Upload button
    tk.Button(root, text="Upload Dataset", command=update_forecasts, bg='#90ee90', font=("Times New Roman", 12, "bold")).pack(pady=10)

    # Forecast model buttons
    model_frame = tk.Frame(root, bg='#f3e8ff')
    model_frame.pack(pady=5)

    for model in ['ARIMA', 'GBDT', 'Random Forest', 'Linear Regression']:
        tk.Button(model_frame, text=model, command=lambda m=model: show_graph(m), bg='#90ee90',
                  font=("Times New Roman", 10, "bold")).pack(side=tk.LEFT, padx=5)

    # Graph buttons
    graph_frame = tk.Frame(root, bg='#f3e8ff')
    graph_frame.pack(pady=10)

    buttons = [
        ("Product Comparison", show_product_comparison),
        ("Sales Heatmap", show_sales_heatmap),
        ("Yearly Boxplot", show_yearly_boxplot),
        ("Model Scores", show_model_performance),
        ("Model Residuals", show_residual_plot)
    ]
    for label, func in buttons:
        tk.Button(graph_frame, text=label, command=func, bg='#90ee90',
                  font=("Times New Roman", 10, "bold")).pack(side=tk.LEFT, padx=5)

    # Forecast plot area
    frame = tk.Frame(root, bg='white')
    frame.pack(pady=20, fill='both', expand=True)

    root.mainloop()

if __name__ == '__main__':
    create_dashboard()
