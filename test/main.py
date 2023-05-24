import pickle

from preprocess import preprocess


# Load the dataset

print("\nLoading dataset ...")
try:
    # Here the filepath to the dataset has to be inserted
    with open("/home/patrick/Desktop/sviat_final_project/catvsnotcat_small.pkl", "rb") as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Dataset file not found.")
    exit()
except Exception as e:
    print("Error loading dataset:", str(e))
    exit()


# Preprocess dataset
data_fold_dict = preprocess(dataset)

"""
Input layer:
128x128 = 16384 neurons for the input layer
Output layer:
1 single neuron
"""

