import pickle


def load(filepath: str) -> list[dict]:
    """
    This function loads the dataset from a given filepath

    Parameters:
    -----------
    filepath: str
        filepath to the dataset

    Returns:
    ----------------------
    dataset: list(dict)
        dataset containing all labeled images to train and test the network
    """
    print("\nLoading dataset ...")

    try:
        # Here the filepath to the dataset has to be inserted!!!
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Dataset file not found.")
        exit()
    except Exception as e:
        print("Error loading dataset:", str(e))
        exit()

    return dataset
