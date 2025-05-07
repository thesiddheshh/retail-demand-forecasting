import matplotlib.pyplot as plt

def plot_forecast(train, test, predictions, title="Forecast"):
    """
    Plot the forecasted results.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['monthly_sales'], label='Train')
    plt.plot(test.index, test['monthly_sales'], label='Test')
    plt.plot(test.index, predictions, label='Forecast')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Monthly Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{title.replace(' ', '_').lower()}.png")
    plt.show()

def compare_forecasts(actual, arima_preds, lstm_preds):
    """
    Compare ARIMA and LSTM forecasts.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(actual.index, arima_preds, label='ARIMA Forecast')
    plt.plot(actual.index, lstm_preds, label='LSTM Forecast')
    plt.title("Model Comparison")
    plt.xlabel('Date')
    plt.ylabel('Monthly Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison.png")
    plt.show()
