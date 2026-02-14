import math
import random
class Element:
    def __init__(self, data, operands: tuple = (), operation: str = None, label=""):
        self.data = data
        self.operands = set(operands)
        self.operation = operation
        self.gradient = 0.0
        self.backpropagation = lambda: None
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.data + other.data, 
            operands=(self, other), 
            operation="+", 
            label=f"{self.label} + {other.label}"
            )

        def backward():
            self.gradient += 1.0 * output.gradient
            other.gradient += 1.0 * output.gradient

        output.backpropagation = backward
        return output

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.data * other.data,
            operands=(self, other),
            operation="*",
            label=f"{self.label} * {other.label}"
        )

        def backward():
            self.gradient += other.data * output.gradient
            other.gradient += self.data * output.gradient

        output.backpropagation = backward
        return output

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.data - other.data,
            operands=(self, other),
            operation="-",
            label=f"{self.label} - {other.label}"
        )
        def backward():
            self.gradient += 1.0 * output.gradient
            other.gradient += -1.0 * output.gradient

        output.backpropagation = backward
        return output

    def __rsub__(self, other):
        return Element(other) - self

    def __truediv__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.data / other.data,
            operands=(self, other),
            operation="/",
            label=f"{self.label} / {other.label}"
        )

        def backward():
            self.gradient += 1/other.data * output.gradient
            other.gradient += (-self.data/(other.data**2)) * output.gradient

        output.backpropagation = backward
        return output

    def __pow__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.data**other.data,
            operands=(self, other),
            operation="**",
            label=f"{self.label} ** {other.label}"
        )
        def backward():
            self.gradient += other.data * self.data**(other.data-1) * output.gradient
            if self.data > 0:
                other.gradient += output.data * math.log(self.data) * output.gradient

        output.backpropagation = backward
        return output

    def tanh(self):
        function = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        output = Element(function, operands=(self, ), label="tanh")

        def backward():
            self.gradient += (1-function**2) * output.gradient

        output.backpropagation = backward
        return output

    def backprop(self):
        graph = []
        visited = set()
        self.gradient = 1.0

        def computational_graph(node):
            if node not in visited:
                visited.add(node)
                for operand in node.operands:
                    computational_graph(operand)
                graph.append(node)

        computational_graph(self)
        
        for node in reversed(graph):
            node.backpropagation()

    def __repr__(self):
        result = f"Label: {self.label}, Value: {self.data}"
        if self.operands != set():
            result += f", Operands: {self.operands}"
        if self.operation is not None:
            result += f", Operation: {self.operation}"
        return result


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Element(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Element(random.uniform(-1,1))

    def __call__(self, x):
        z = sum(xi*wi for wi, xi in zip(self.w, x)) + self.b
        a = z.tanh()
        return a

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [parameters for neuron in self.neurons for parameters in neuron.parameters()]


class NN:
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outputs))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameters for layer in self.layers for parameters in layer.parameters()]


if __name__ == "__main__":

    model = NN(3, [4,4,1])
    X = [
        [2.0,1.5,-1.0],
        [2.0, -1.0, 0.5],
        [1.0,1.0,-1.0],
        [2.1, 1.4, -1.2]
    ]
    y = [0.5,-1.0,-0.8,0.25]

    epochs = 100
    for epoch in range(epochs):
        ypred = [model(x) for x in X]
        loss = sum((y_pred - y)**2 for y, y_pred in zip(y, ypred))

        for parameter in model.parameters():
            parameter.gradient = 0.0

        loss.backprop()

        for parameter in model.parameters():
            parameter.data += -0.05 * parameter.gradient
        
        print(epoch, loss.data)
