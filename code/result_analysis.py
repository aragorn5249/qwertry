# here goes the code for the result analysis

def result_analysis(parameters, X, Y):
    """
    Analyze the results of the trained model

    Arguments:
    parameters -- a dictionary containing the trained parameters
    X -- input data of shape (input size = 2, number of examples = 200)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    predictions -- vector of predictions of the model (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions
