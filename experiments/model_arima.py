from statsmodels.tsa.statespace.sarimax import SARIMAX

def train(train_series):
    """
    Train an ARIMA model on the training data.
    """
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit
