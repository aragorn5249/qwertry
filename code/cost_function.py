# here goes the code for the cost function

def compute_cost(Yhat, Y):
    """
    Implement the cross-entropy cost function
 
    Arguments:
    Yhat -- probability vector corresponding to the label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
 
    Returns:
    cost -- cross-entropy cost
    """
 
    m = Y.shape[1]
 
        # Compute loss from AL and Y.
    logprobs = np.dot(Y, np.log(Yhat).T) + np.dot((1-Y), np.log(1-Yhat).T)
 
    cost = (-1./m) * logprobs 
 
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
 
    return cost
