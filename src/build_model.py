from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model for binary classification.

    Args:
        input_shape (tuple): Shape of input data, e.g. (60, 1)

    Returns:
        tensorflow.keras.Model: Compiled LSTM model.
    """
    model = Sequential()

    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model