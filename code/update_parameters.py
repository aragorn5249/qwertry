import numpy as np


def update_parameters(
    parameters: dict, grads: np.ndarray, LEARNING_RATE: float
) -> dict:
    """
    Update parameters using gradient descent

    Parameters:
    -----------
    parameters : dict
        dictionary containing your parameters
    grads : np.ndarray
        python dictionary containing your gradients, output of L_model_backward
    LEARNING_RATE : float
        user defined learning rate

    Returns:
    --------
    parameters : dict
        python dictionary containing your updated parameters
            parameters["W" + str(l)] = ...
            parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - LEARNING_RATE * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - LEARNING_RATE * grads["db" + str(l + 1)]
        )

    return parameters
