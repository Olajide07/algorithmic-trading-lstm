import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_close_prices(data):
    """
    Extract and scale closing prices between 0 and 1.
    """
    close_prices = data[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    return scaled_prices, scaler


def create_sequences(scaled_prices, window_size=60):
    """
    Create 60-day input sequences and binary UP/DOWN labels.

    Label:
        1 = next-day price is UP
        0 = next-day price is DOWN or unchanged
    """
    X = []
    y = []

    for i in range(window_size, len(scaled_prices) - 1):
        X.append(scaled_prices[i - window_size:i])

        if scaled_prices[i + 1] > scaled_prices[i]:
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y)


def train_test_split_time_series(X, y, train_ratio=0.8):
    """
    Split time-series data without shuffling.
    """
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test