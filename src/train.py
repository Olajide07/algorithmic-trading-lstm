def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Train the LSTM model.

    Args:
        model: Compiled Keras model.
        X_train: Training features.
        y_train: Training labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.

    Returns:
        History object from Keras training.
    """
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    return history