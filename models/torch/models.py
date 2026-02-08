import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_size: int, num_neurons: int, activation_function=None):
        super().__init__()
        self.linear = nn.Linear(input_size, num_neurons)
        self.activation_function = activation_function

        # He initialization
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        if self.activation_function:
            return self.activation_function(z)
        return z


class SequentialModel(nn.Module):
    def __init__(self, loss_function, layers: list[LinearLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.loss_function = loss_function

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, X, y, optimizer, epochs=10, batch_size=32):
        m = X.shape[0]
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            loss_history.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1} loss: {avg_epoch_loss}")

        return loss_history
