import numpy as np

def initialize_parameters():
    """
Function is responsible for initializing the parameters (weights and biases) of a neural network. 
In this case, the function is designed for a neural network 
with a single hidden layer 
and an input layer of size 16,384 neurons.

    """
    parameters = {}
    
    parameters['W1'] = np.random.randn(1, 16384) * 0.01
    parameters['b1'] = np.zeros((1, 1))
    
    return parameters

def sigmoid(Z):
    """
    Apply the sigmoid activation function to the input Z.
    """
    A = 1 / (1 + np.exp(-Z))
    return A

def linear_activation_forward(A_prev, W, b):
    """
    Perform the forward propagation step for a single layer.
    Cache needed for the storing of valuse fo BP
    """
    Z = np.dot(W, A_prev) + b
    A = sigmoid(Z)
    cache = (A_prev, W, b)
    
    return A, cache

def L_layer_model(X, Y, learning_rate, num_iterations, print_cost=False):
    """
    Train the neural network with L layers.
    """
    np.random.seed(1)
    costs = []
        
    parameters = initialize_parameters()
    
    for i in range(num_iterations):
        
        AL, cache = linear_activation_forward(X, parameters['W1'], parameters['b1'])
        
        """
        Compute cost
        """
        cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        """
        Backward propagation
        """
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ = dAL * AL * (1 - AL)
        dW = np.dot(dZ, cache[0].T) / X.shape[1]
        db = np.mean(dZ, axis=1, keepdims=True)
        
        """
        Parameters update
        """
        parameters['W1'] -= learning_rate * dW
        parameters['b1'] -= learning_rate * db
        
        """
        This is for the cost print every 100 itterations.
        """
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    return parameters

"""
The code represents a neural network with a single hidden layer consisting of fully connected neurons. 
In this implementation, the input layer has 16,384 neuronsThe output layer is a single neuron.
"""
