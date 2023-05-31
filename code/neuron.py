import numpy as np

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.parameters = {}
        """
        I did as you advised me. so this part is where "claass" crated with the proper data. 
        self.parameters is an empty dixtionary, but it should get data during the training
        """
    
    def initialize_parameters(self):
        self.parameters['W1'] = np.random.randn(1, 16384) * 0.01
        self.parameters['b1'] = np.zeros((1, 1))
    """
    initializes the weight W1 with 16384 neurons and the bias b1 with zeros. In theory W1 could be -  np.random.randn(1, self.input_size) * 0.01
    these parameters will be used during the training or forward propagation of the neural network.
    """
    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def linear_activation_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        A = self.sigmoid(Z)
        cache = (A_prev, W, b, Z, A)
        
        return A, cache
    """
    This method performs the forward propagation step for a single layer in the neural network
    """
    def compute_cost(self, AL, Y):
        cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return cost
    """
    It calculates cost, which should show how well the neural network's . 
    The goal of training the network is to minimize this cost function through optimization algorithms.
    """
    def backward_propagation(self, dAL, cache):
        A_prev, W, b, Z, A = cache
        dZ = dAL * A * (1 - A)
        dW = np.dot(dZ, A_prev.T) / A_prev.shape[1]
        db = np.mean(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    """
    This BP function calculates the gradients of the cost function with respect to the parameters of the neural network using the chain rule. 
    These gradients are important for updating the parameters during the optimization process, such as in gradient descent or other optimization algorithms.
    """
    def update_parameters(self, dW, db, learning_rate):
        self.parameters['W1'] -= learning_rate * dW
        self.parameters['b1'] -= learning_rate * db
    """
    function applies the gradient descent optimization algorithm to update the parameters of the neural network
    """
    def train(self, X, Y, learning_rate, num_iterations, print_cost=False):
        np.random.seed(1)
        self.initialize_parameters()
        costs = []

        for i in range(num_iterations):
            AL, cache = self.linear_activation_forward(X, self.parameters['W1'], self.parameters['b1'])
            cost = self.compute_cost(AL, Y)
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
            dA_prev, dW, db = self.backward_propagation(dAL, cache)
            self.update_parameters(dW, db, learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
        
        return self.parameters
    """
    This suppose to be training process. but not sure if it is ok
    """
