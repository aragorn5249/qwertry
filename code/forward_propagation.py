import numpy as np


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple:
    """
    Implement the linear part of a layer's forward propagation.

    Parameters:
    -----------
    A : np.ndarray
        activations from previous layer (or input data): (size of previous layer, number of examples)
    W : np.ndarray
        weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b : np.ndarray
        bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    --------
    Z : np.ndarray
        input of the activation function, also called pre-activation parameter
    cache : list
        dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z: np.ndarray = np.dot(W, A) + b

    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


def sigmoid(Z: np.ndarray) -> tuple:
    """
    Implements the sigmoid activation in numpy

    Parameters:
    -----------
    Z : np.ndarray
        Output of the linear layer

    Returns:
    --------
    A : np.ndarray
        the output of the activation function, also called the post-activation value
    cache : list
        dictionary containing "linear_cache" and "activation_cache";
            -> stored for computing the backward pass efficiently
    """
    A: np.ndarray = 1 / (1 + np.exp(-Z))
    cache: list = Z

    return A, cache


def relu(Z: np.ndarray) -> tuple:
    """
    Implement the RELU function.

    Parameters:
    -----------
    Z : np.ndarray
        Output of the linear layer

    Returns:
    --------
    A : np.ndarray
        the output of the activation function, also called the post-activation value
    cache : list
        dictionary containing "linear_cache" and "activation_cache";
            -> stored for computing the backward pass efficiently
    """
    A: np.ndarray = np.maximum(0, Z)

    assert A.shape == Z.shape

    cache: list = Z
    return A, cache


def linear_activation_forward(
    A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str
) -> tuple:
    """
    Implement the forward propagation for the LINEAR activation layer

    Parameters:
    -----------
    A_prev : np.ndarray
        activations from previous layer (or input data): (size of previous layer, number of examples)
    W : np.ndarray
        weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b : np.ndarray
        bias vector, numpy array of shape (size of the current layer, 1)
    activation : str
        the activation function to be used in this layer ("sigmoid" or "relu")

    Returns:
    --------
    A : np.ndarray
        the output of the activation function, also called the post-activation value
    cache : list
        dictionary containing "linear_cache" and "activation_cache";
            -> stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache: list = (linear_cache, activation_cache)

    return A, cache


def perform_forward_propagation(images: np.ndarray, parameters: dict):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Parameters:
    -----------
    images : np.ndarray[float]
        numpy array containing all images -> shape (input size, number of examples)
    parameters : dict
        dictionary containing the values for the network parameters learnt by the model. They are used later to predict the test images.

    Returns:
    --------
    Y_hat : np.ndarray
        last post-activation value
    caches : list
        list of caches containing:
        - every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
        - the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches: list = []
    A: np.ndarray = images
    number_of_layers: int = (
        len(parameters) // 2
    )  # number of layers in the neural network

    # Implement [LINEAR ->; RELU]*(L-1). Add "cache" to the "caches" list.
    for i in range(1, number_of_layers):
        A_prev: np.ndarray = A
        w_l: np.ndarray = parameters["W" + str(i)]
        b_l: np.ndarray = parameters["b" + str(i)]
        A, cache = linear_activation_forward(A_prev, w_l, b_l, activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    w_L: np.ndarray = parameters["W" + str(number_of_layers)]
    b_L: np.ndarray = parameters["b" + str(number_of_layers)]
    Y_hat, cache = linear_activation_forward(A, w_L, b_L, activation="sigmoid")
    caches.append(cache)

    assert Y_hat.shape == (1, images.shape[1])

    return Y_hat, caches
