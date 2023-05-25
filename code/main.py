import pickle

from analysis import analyze
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
