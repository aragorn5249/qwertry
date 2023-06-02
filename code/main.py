import pickle

from code.dataset_analysis import analyze
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
dataset_fold_dict:dict[str, list[dict]] = preprocess(dataset)

"""
Input layer:
128x128 = 16384 neurons for the input layer
Output layer:
1 single neuron
"""

# Defining neural network structure
### CONSTANTS DEFINING THE MODEL ####
n_x = train_set_x_flatten.shape[0]     # size of input layer
n_y = 1  # size of output layer, will be 0 or 1
    # we define a neural network with total 5 layers, x, y and 3 hidden:
    # the first hidden has 20 units, second has 7 units and third has 5
nn_layers = [n_x, 20, 7, 5, n_y]  # length is 5 (layers)
 
nn_layers
