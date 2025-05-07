import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data(series, look_back=12):
    """
    Prepare data for LSTM model.
    """
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_series) - look_back):
        X.append(scaled_series[i:i+look_back])
        y.append(scaled_series[i+look_back])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def train(X, y):
    """
    Train an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    return model
