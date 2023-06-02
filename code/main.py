import pickle

from dataset_analysis import analyze
from preprocessing import preprocess


# Load dataset
print("\nLoading dataset ...")
try:
    # Here the filepath to the dataset has to be inserted!!!
    with open(
        "/home/patrick/Desktop/sviat_final_project/catvsnotcat_small.pkl", "rb"
    ) as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Dataset file not found.")
    exit()
except Exception as e:
    print("Error loading dataset:", str(e))
    exit()

# Analyze dataset
analyze(dataset)

# Preprocess dataset
training_dataset, test_dataset = preprocess(dataset)

"""
Input layer:
128x128 = 16384 neurons for the input layer
Output layer:
1 single neuron
"""

# Defining neural network structure
# size of input layers
INPUT_LAYER_SIZE:int = training_dataset[0]["X"].size
# size of output layer -> 1 since we want to get only a probablity value
OUTPUT_LAYER_SIZE:int = 1
# length of array corresponds to the number of hidden layers (here 3 hidden layers) with the values being the respective sizes
HIDDEN_LAYER_SIZES:list[int] = [20, 7, 5] 

nn_layers = [INPUT_LAYER_SIZE] + HIDDEN_LAYER_SIZES + [OUTPUT_LAYER_SIZE]
# print(nn_layers)
print(f"\nNeural network properties:")
print(f"size of input_layer: {INPUT_LAYER_SIZE}")
[print(f"size of hidden_layer_{i+1}: {layer_size}") for i, layer_size in enumerate(HIDDEN_LAYER_SIZES)]
print(f"size of output_layer: {OUTPUT_LAYER_SIZE}")


