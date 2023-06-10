import numpy as np
from forward_propagation import perform_forward_propagation


def predict(
    images: np.ndarray, labels: np.ndarray, parameters: dict, name: str
) -> tuple:
    """
    This function is used to predict the results of a  L-layer neural network.

    Parameters:
    -----------
    images : np.ndarray
        data set of examples you would like to label
    parameters : dict
        parameters of the trained model
    name: str
        name of the dataset to be predicted

    Returns:
    --------
    accuracy : float
        ratio of correctly classified images
    predictions : np.ndarray
        predictions for the given dataset X
    """
    number_of_images: int = images.shape[0]
    predictions = np.zeros((1, number_of_images))

    # Forward propagation
    probas, _ = perform_forward_propagation(np.transpose(images), parameters)

    # convert probs to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0

    # Calculate accuracy
    accuracy: float = np.round(np.sum((predictions == labels) / number_of_images), 3)

    # print results
    print(f"- accuracy of {name}: {accuracy}")

    return accuracy, predictions
