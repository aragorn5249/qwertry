import pathlib
import pickle
import numpy as np

from tqdm import tqdm

from initialize_parameters import initialize_parameters
from forward_propagation import perform_forward_propagation
from cost_function import compute_cost
from backward_propagation import perform_backward_propagation
from update_parameters import update_parameters


def create_and_train_neural_network(
    images: np.ndarray[float],
    labels: np.ndarray,
    layer_dimensions: list[int],
    LEARNING_RATE: float,
    NUMBER_OF_ITERATIONS: int,
    OUTPUT_DIRECTORY: pathlib.Path,
    STORE_PARAMETERS_AND_RESULT: bool,
) -> tuple():
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Parameters:
    -----------
    images : np.ndarray
        image matrix of shape (number of examples, number_pixels_heigt * number_pixels_width)
    labels : np.ndarray
        label vector of the corresponding images (containing 0 if cat, 1 if non-cat), of shape (1, number of images)
    layer_dimensions : list[int]
        list containing the input size and each layer size, of length (number of layers + 1).
    LEARNING_RATE : float
        learning rate of the gradient descent update rule
    NUMBER_OF_ITERATIONS : int
        number of iterations of the optimization loop
    OUTPUT_DIRECTORY : pathlib.Path
        path to the directory where the results are stored
    STORE_PARAMETERS_AND_RESULT : bool
        defines whether the parameters of the network are stored in pickle file

    Returns:
    --------
    costs : list[float]
        list containing the cost (=cross-entropy) of every iteration
    parameters : dict
        dictionary containing the values for the network parameters learnt by the model. They are used later to predict the test images.
    """
    costs: list[float] = []

    # Parameters initialization.
    parameters: dict = initialize_parameters(layer_dimensions)

    # Loop (gradient descent)
    print(f"\nTraining network ...")
    for _ in tqdm(range(0, NUMBER_OF_ITERATIONS)):
        # Forward propagation
        AL, caches = perform_forward_propagation(np.transpose(images), parameters)

        # Compute cost
        cost: float = compute_cost(AL, labels)

        # Backward propagation
        grads: np.ndarray = perform_backward_propagation(AL, labels, caches)

        # Update parameters
        parameters: dict = update_parameters(parameters, grads, LEARNING_RATE)

        # Append costs
        costs.append(cost)

    # Store parameter values in pickle file
    if STORE_PARAMETERS_AND_RESULT:
        with open(f"{OUTPUT_DIRECTORY}/parameters.pickle", "wb") as file:
            pickle.dump(parameters, file, pickle.HIGHEST_PROTOCOL)

    return costs, parameters
