### Structure

```
ml-projects/
├── api/
│   └── api.py                 # API testing point
│
├── ml_models/
│   ├── src/
│   │   ├── backprop_engine.py # Backprop engine from     	scratch @karpathy
│   │   │   
│   │   └── main.py            # Neural networks from scratch
│   │                            (activations, losses, layers, backprop and mini-batch training)
│   └── torch/
│       └── models.py          # Neural network models with PyTorch
│
├── system_dynamics/
│   └── state.py               # Dynamic systems modeling with states, dimensions and actions
│
├── utils/
│   ├── backprop/
│   │   └── backprop.ipynb     # Backpropagation experiments and derivative calculations
│   ├── nlp/
│   │   └── tokenizer.py       # BPE (Byte Pair Encoding) tokenizer
│   ├── rnns/
│   │   └── recurrent_nn.ipynb # Recurrent neural network with PyTorch
│   └── tensorflow_testing/
│       └── simple_classification_minst.ipynb  # Simple MNIST classification with TensorFlow/Keras
│
├── datasets/                  # Datasets (CSV files)
├── tests/
│   └── test_function.py       # Unit tests
├── main.py                    # Entry point
└── requirements.txt           # Project dependencies
```
