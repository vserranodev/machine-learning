import torch
import torch.nn as nn
import numpy as np
import enum

class ActivationFunction(enum):
    ####To access the proper function --> ActivationFunction.ReLU.value()
    ####If we access just ActivationFunction.ReLU --> we access just the enum object
    #### If we access ActivationFunction.ReLU.name() --> We access the name --> ReLU
    ReLU: nn.ReLU
    Tanh: nn.Tanh
    Sigmoid: nn.Sigmoid
    LeakyReLU: nn.LeakyReLU

class Optimizers(enum):
    Adam: torch.optim.Adam
    SGD: torch.optim.SGD

class LinearModel(nn.Module):
    def __init__(self, input_size: tuple, output_size:tuple = (1,1), num_layers=1, activation_function=ActivationFunction.ReLU.value()):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.weights = np.random.randn(*input_size) * 0.01
        self.bias = np.ones((input_size[0],1))

    def forward(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        A = self.activation_function(Z)
        return A

    def MeanSquaredError(self, y_pred, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        errors = np.mean((y_pred - y)**2)

        return errors
        
    ###Asumming training examples is always dim=0
    ###COMING SOON --> COMPUTATIONAL GRAPH FROM SCRATCH
    #### Preferrable to enter data in batchsize --> Preferrably as DataLoaders in torch
    def train(self,lossfun, learning_rate, optimizer, data, epochs=100):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X,y in data:
                optimizer.zero_grad()
                y_pred = self(X)
                loss = lossfun(y_pred, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch} | Loss: {epoch_loss}")


if __name__ == "__main__":
    X = None
    model = LinearModel(input_size=(20,10))
    model.train(lossfun=model.MeanSquaredError, epochs=100, optimizer = Optimizers.Adam.value(model.parameters(), lr=0.01), data=data)
    y_preds = model(X)
    print(y_preds)






