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
        lambda z: np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
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
    def __init__(self, input_size, num_neurons: int, activation_function: ActivationFunction):
        self.input = None
        self.input_size: tuple = input_size
        self.num_neurons = num_neurons
        self.activation_function = activation_function.function
        self.activation_function_derivative = activation_function.derivative
        self.weights = np.random.randn(input_size, num_neurons) * 0.01
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
        x = X
        for layer in self.layers:
            layer.input = x
            x = layer()
        return x

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
        epoch_idx = 0


        for epoch in range(epochs):
            epoch_idx += 1
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, m, batch_size):
                batch_loss = 0.0
                X_batch = X[:, i:i + batch_size]
                y_batch = y[:, i: i + batch_size]
                y_pred = self.forward_propagation(X_batch)
                batch_loss = self.get_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1

                self.backward_propagation(y_pred, y_batch, learning_rate)

            avg_epoch_loss = epoch_loss / num_batches
            loss_history.append(avg_epoch_loss)
            print(f"Epoch {epoch_idx} loss: {avg_epoch_loss}")


if __name__ == "__main__":
    
    ## Testing it out
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    data = fetch_california_housing()
    X, y = data.data, data.target # X: (20640, 8), y: (20640,)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # (20640, 8)

    # Transposing to get (features, examples)
    X_scaled = X_scaled.T 
    y = y.reshape(1, -1) 

    # Divide into train and test
    # Note: train_test_split likes (examples, features), so we transpose and then we change again
    X_train, X_test, y_train, y_test = train_test_split(X_scaled.T, y.T, test_size=0.2, random_state=42)

    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T

    num_features = X_train.shape[0] # Esto debería ser 8

    capa_oculta = LinearLayer(
        input_size=num_features, 
        num_neurons=10, 
        activation_function=ActivationFunction.ReLU
    )

    # Last layer has 10 neurons,, so this layer receives 10
    capa_salida = LinearLayer(
        input_size=10, 
        num_neurons=1, 
        activation_function=ActivationFunction.ReLU 
    )

    # Train
    model = LinearModel(loss_function=LossFunction.MSE, layers=[capa_oculta, capa_salida])

    print(f"Iniciando entrenamiento con {X_train.shape[1]} muestras...")
    history = model.train(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)

    # Evaluation
    pred = model.forward_propagation(X_test[:, :5])

    print("\n" + "="*35)
    print(f"{'REAL':<15} | {'PREDICCIÓN':<15}")
    print("-" * 35)

    for r, p in zip(y_test[0, :5], pred.flatten()):
        print(f"{r:<15.4f} | {p:<15.4f}")
    print("="*35)