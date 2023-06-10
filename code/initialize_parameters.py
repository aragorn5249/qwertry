#  FUNCTION: initialise_parameters
import numpy as np


def initialize_parameters(layer_dimensions: list[int]) -> dict:
    """
    Initialize weight matrices with all zeros and bias vectors with random numbers between 0 and 1

    Parameters:
    -----------
    layer_dimensions : list[int]
        python array (list) containing the dimensions of each layer in our network

    Returns:
    --------
    parameters : dict
        python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
            Wl: weight matrix of shape (layer_dimensions[l], layer_dimensions[l-1])
            bl: bias vector of shape (layer_dimensions[l], 1)
    """
    print("\nInitializing weight matrices and bias vectors ...")
    parameters = {}
    L = len(layer_dimensions)  # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layer_dimensions[l], layer_dimensions[l - 1]
        ) / np.sqrt(layer_dimensions[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

        # unit tests
        assert parameters["W" + str(l)].shape == (
            layer_dimensions[l],
            layer_dimensions[l - 1],
        )
        assert parameters["b" + str(l)].shape == (layer_dimensions[l], 1)

    return parameters
