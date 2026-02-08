import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    ### Activation function and derivative in tuple form for backprop
    ### self.activation_function = self.activation_function.value[0]
    ### self.activation_function_derivative = self.activation_function.value[1]

    ReLU = (
        #Activation function
        lambda z: np.maximum(0, z),
        #Derivative of activation function
        lambda z: (z > 0).astype(float)
    )
    Tanh = (
        #Activation function
        lambda z: np.tanh(z),
        #Derivative of activation function
        lambda z: 1 - np.tanh(z)**2
        )
    Sigmoid = (
        #Activation function
        lambda z: 1 / (1 + np.exp(-z)),
        #Derivative of activation function
        lambda z: (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
        )
    LeakyReLU = (
        #Activation function
        lambda z: np.where(z > 0, z, z * 0.01),
        #Derivative of activation function
        lambda z: np.where(z > 0, 1, 0.01)
        )

    @property 
    def function(self):
        return self.value[0]

    @property 
    def derivative(self):
        return self.value[1]


class LossFunction(Enum):
    ## TO DO --> Implementing regularization term to add to lossfun

    MSE = (
        #Loss function
        lambda y_pred, y: 1/2 * ((y_pred - y)**2),
        #Derivative of loss function
        lambda y_pred, y: y_pred - y
    )

    RMSE = (
        #Loss function
        lambda y_pred, y: np.sqrt(1/2 * ((y_pred - y)**2)),
        #Derivative of loss function
        lambda y_pred, y, epsilon=1e-8: (y_pred - y) / (2 * np.sqrt((1/2 *  ( (y_pred - y) **2 ))  + epsilon) )
    )
    MAE = (
        #Loss function
        lambda y_pred, y: np.abs(y_pred - y),
        #Derivative of loss function
        lambda y_pred, y: np.sign(y_pred - y)
    )

    @property
    def loss(self):
        return self.value[0]

    @property
    def derivative(self):
        return self.value[1]


class LinearLayer():
    ##Assuming first dim as features
    def __init__(self, input, num_neurons: int, activation_function: ActivationFunction):
        self.input = input
        self.input_size: tuple = (input.shape[0], input.shape[1])
        self.num_neurons = num_neurons
        self.activation_function = activation_function.function
        self.activation_function_derivative = activation_function.derivative
        self.weights = np.random.randn(input.shape[0], num_neurons) * 0.01
        self.bias = np.zeros((self.num_neurons, 1))
        self.Z = None
        self.A = None

    def __call__(self):
        Z = np.dot(self.weights.T, self.input) + self.bias
        self.Z = Z
        A = self.activation_function(Z)
        self.A = A
        return A

    def backpropagation(self, learning_rate, dA):
        # Number of training examples so we can divide by these
        m = self.input.shape[1]
        # Derivative of activation function with respect to Z --> dZ = dL/dA * dA/dZ
        dZ = self.activation_function_derivative(self.Z) * dA
        # Derivative of Z respecto to the weights (self.input) * dZ --> dW = dL/dA * dA/dZ * dZ/dW
        dW = np.dot(self.input, dZ.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        ### To know how much the inputs from the previous layer contributed to the error of this layer, we must:
        ### check the derivative of Z with respect to those inputs from previous layer
        ### derivative of Z with respect to input X: dZ/dX --> dL/dA * dA/dZ * dZ/dX
        ### (Features, num_neurons) x (num_neurons, m) = (Features, m)
        prev_layer_dA = np.dot(self.weights, dZ)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return prev_layer_dA


class LinearModel():
    def __init__(self, 
        loss_function: LossFunction, 
        layers: list[LinearLayer]):

        self.layers = layers
        self.loss_function = loss_function.loss
        self.loss_function_derivative = loss_function.derivative

    def forward_propagation(self, X):
        input = X
        for layer in self.layers:
            layer.input = input
            input = layer()
            if layer == self.layers[-1]:
                output = input
                return output

    def get_loss(self, y_pred, y):
        individual_losses = self.loss_function(y_pred, y)
        return np.mean(individual_losses)

    def backward_propagation(self,y_pred, y, learning_rate):
        dA = self.loss_function_derivative(y_pred, y)
        #y_pred in this function is just the output A of the last layer
        for layer in reversed(self.layers):
            dA = layer.backpropagation(learning_rate, dA)

    def train(self, X, y, epochs=10, learning_rate=0.01, batch_size=64):
        #Number of training examples
        m = X.shape[1]
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, m, batch_size):
                batch_loss = 0.0
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i: i + batch_size]
                y_pred = self.forward_propagation(X_batch)
                loss = self.get_loss(y_pred, y_batch)

                batch_loss += loss
                epoch_loss += batch_loss
                num_batches += 1

                self.backward_propagation(y_pred, y_batch, learning_rate)

            avg_epoch_loss = epoch_loss / num_batches
            loss_history.append(avg_epoch_loss)
            print(f"Epoch loss: {avg_epoch_loss}")


if __name__ == "__main__":
    pass



