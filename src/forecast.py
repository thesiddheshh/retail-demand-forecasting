import numpy as np

def predict(model, periods):
    """
    Generate forecasts using the ARIMA model.
    """
    forecast = model.forecast(steps=periods)
    return forecast

def predict_lstm(model, train_series, scaler, look_back=12, n_future=12):
    """
    Generate forecasts using the LSTM model.
    """
    input_seq = train_series[-look_back:].values.reshape(-1, 1)
    input_seq = scaler.transform(input_seq)
    predictions = []
    for _ in range(n_future):
        input_reshaped = input_seq.reshape(1, look_back, 1)
        pred = model.predict(input_reshaped, verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred, axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions
