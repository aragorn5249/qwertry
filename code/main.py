import pathlib
import pickle
import matplotlib.pyplot as plt
import numpy as np

from functools import reduce
from load_dataset import load
from analyze_dataset import analyze
from preprocessing import preprocess
from neural_network import create_and_train_neural_network
from performance_check import predict


# User defined parameters
FILEPATH: str = "/home/patrick/Desktop/sviat_final_project/catvsnotcat_small.pkl"  # filepath to the dataset to be imported
DESIRED_IMAGE_SIZE: tuple[int] = (64,64)        # greyscale image size to which the all images are converted fto get a unitary input size
TRAIN_NETWORK: bool = False                      # specifiy if network shall be trained -> if False, whole dataset gets only tested and no result file will be created!
if TRAIN_NETWORK:
    STORE_PARAMETERS_AND_RESULT: bool = False   # specify if user defined parameters, network parameters and result shall be stored in result.txt file
    HIDDEN_LAYER_SIZES: list[int] = [20, 7, 5]  # length of array corresponds to the number of hidden layers (here 3 hidden layers) with the values being the respective sizes
    NUMBER_OF_ITERATIONS: int = 3000            # number of training iterations
    LEARNING_RATE: float = 0.0075               # step size the grandient descent is moving forward -> has to be a value in the range of [0,1]

# Type-check user defined parameters
assert isinstance(FILEPATH, str)
assert isinstance(DESIRED_IMAGE_SIZE, tuple)
assert isinstance(TRAIN_NETWORK, bool)
if TRAIN_NETWORK:
    assert isinstance(STORE_PARAMETERS_AND_RESULT, bool)
    assert isinstance(HIDDEN_LAYER_SIZES, list)
    assert isinstance(NUMBER_OF_ITERATIONS, int)
    assert isinstance(LEARNING_RATE, float)
    assert 0 < LEARNING_RATE < 1

# Global constants
OUTPUT_DIRECTORY: pathlib.Path = pathlib.Path(__file__).parent.parent.resolve()
if TRAIN_NETWORK:
    INPUT_LAYER_SIZE: int = reduce(lambda a, b: a * b, DESIRED_IMAGE_SIZE)
    OUTPUT_LAYER_SIZE: int = 1
    NEURAL_NETWORK_LAYERS: list[int] = (
        [INPUT_LAYER_SIZE] + HIDDEN_LAYER_SIZES + [OUTPUT_LAYER_SIZE]
    )

# Print user defined variables and global constants
print("\nUser defined parameters")
print(f"filepath to dataset: {FILEPATH}")
print(f"desired image size:{DESIRED_IMAGE_SIZE}")
print(f"train:{TRAIN_NETWORK}")
if TRAIN_NETWORK:
    print(f"store parameter values in file:{STORE_PARAMETERS_AND_RESULT}")
    print(f"number of iterations: {NUMBER_OF_ITERATIONS}")
    print(f"learning rate: {LEARNING_RATE}")

if TRAIN_NETWORK:
    print(f"\nNeural network properties:")
    print(f"size of input_layer: {INPUT_LAYER_SIZE}")
    [
        print(f"size of hidden_layer_{i+1}: {layer_size}")
        for i, layer_size in enumerate(HIDDEN_LAYER_SIZES)
    ]
    print(f"size of output_layer: {OUTPUT_LAYER_SIZE}")


# Load dataset
dataset: list[dict] = load(FILEPATH)


# Analyze dataset
analyze(dataset)


# Preprocess dataset if network is to be trained
if TRAIN_NETWORK:
    (
        training_dataset_images,
        training_dataset_labels,
        test_dataset_images,
        test_dataset_labels,
    ) = preprocess(dataset, DESIRED_IMAGE_SIZE, TRAIN_NETWORK)
else:
    (
        test_dataset_images,
        test_dataset_labels,
    ) = preprocess(dataset, DESIRED_IMAGE_SIZE, TRAIN_NETWORK)


# Create and train network if network is to be trained
if TRAIN_NETWORK:
    costs, parameters = create_and_train_neural_network(
        training_dataset_images,
        training_dataset_labels,
        NEURAL_NETWORK_LAYERS,
        LEARNING_RATE,
        NUMBER_OF_ITERATIONS,
        OUTPUT_DIRECTORY,
        STORE_PARAMETERS_AND_RESULT,
    )
else:
    with open(f"{OUTPUT_DIRECTORY}/parameters.pickle", "rb") as file:
        parameters = pickle.load(file)


# Check performance of the trained network and store result in text file
print("\nResult:")
if TRAIN_NETWORK:
    accuracy_trainingset, predictions_trainingset = predict(
        training_dataset_images, training_dataset_labels, parameters, "training set"
    )
accuracy_testset, predictions_testset = predict(
    test_dataset_images, test_dataset_labels, parameters, "test set"
)


# Store user defined parameters and result in text file if desired
if TRAIN_NETWORK and STORE_PARAMETERS_AND_RESULT:
    with open(
        f"{OUTPUT_DIRECTORY}/user_defined_parameters_and_result.txt", "w"
    ) as file:
        file.write("User defined parameters:")
        file.write(f"\n- desired image size: {DESIRED_IMAGE_SIZE}")
        file.write(f"\n- input layer size: {INPUT_LAYER_SIZE}")
        file.write(f"\n- hidden layer sizes: {HIDDEN_LAYER_SIZES}")
        file.write(f"\n- output layer size: {OUTPUT_LAYER_SIZE}")
        file.write(f"\n- number of iterations: {NUMBER_OF_ITERATIONS}")
        file.write(f"\n- learning rate: {LEARNING_RATE}")
        file.write("\n\nResult:")
        file.write(f"\n- accuracy trainingset: {accuracy_trainingset}")
        file.write(f"\n- accuracy test set: {accuracy_testset}")


# Display cost evolution if network was trained
if TRAIN_NETWORK:
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title(f"learning rate = {LEARNING_RATE}")
    if STORE_PARAMETERS_AND_RESULT:
        plt.savefig(f"{OUTPUT_DIRECTORY}/cross_entropy.svg")
    plt.show()
