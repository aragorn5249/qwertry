# here goes the code for the cost function
import numpy as np


def compute_cost(Y_hat: np.ndarray, Y: np.ndarray) -> float:
    """
    Implement the cross-entropy cost function

    Parameters:
    -----------
    Y_hat : np.ndarray
        probability vector corresponding to the label predictions, shape (1, number of examples)
    Y : np.ndarray
        true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    --------
    cost : float
        cross-entropy cost
    """
    m = Y.shape[1]

    # Compute loss from AL and Y.
    logprobs = np.dot(Y, np.log(Y_hat).T) + np.dot((1 - Y), np.log(1 - Y_hat).T)

    cost = (-1.0 / m) * logprobs

    cost = np.squeeze(
        cost
    )  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert cost.shape == ()

    return cost
