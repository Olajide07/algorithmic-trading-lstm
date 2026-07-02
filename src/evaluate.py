import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def predict_classes(model, X_test, threshold=0.5):
    """
    Generate binary predictions from model probabilities.
    """
    probabilities = model.predict(X_test)
    predictions = (probabilities > threshold).astype(int)

    return probabilities, predictions


def evaluate_classification(y_test, predictions):
    """
    Evaluate classification performance.
    """
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, matrix, report


def plot_training_history(history):
    """
    Plot training and validation loss.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_confusion_matrix(matrix):
    """
    Plot confusion matrix using matplotlib.
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.show()