import numpy as np
from Layers import Linear
from tensor import Tensor


class NeuralNetworkNumpy:
    def __init__(self):
        self.layer1 = Linear(28*28, 512)
        self.layer2 = Linear(512, 512)
        self.layer3 = Linear(512, 10)

        self.weights = [self.layer1.weights, self.layer2.weights, self.layer3.weights]
        self.biases = [self.layer1.bias, self.layer2.bias, self.layer3.bias]

    def forward(self, x):
        out = x.reshape(x.shape[0], 28*28).view(Tensor)
        out = self.layer1(out)
        out = out.relu()
        out = self.layer2(out)
        out = out.relu()
        out = self.layer3(out)
        return out
