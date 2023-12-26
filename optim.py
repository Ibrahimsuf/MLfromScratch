import numpy as np
class optim:
    def __init__(self, parameters):
        self.weights, self.biases = parameters    
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
            for weight, bias in zip(self.weights, self.biases):
                weight.zero_grad()
                bias.zero_grad()

class SGD(optim):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.learning_rate = learning_rate
    
    def step(self):
        for weights, biases in zip(self.weights, self.biases):
            weights -= weights.gradients * self.learning_rate
            biases -= biases.gradients * self.learning_rate

class Adam(optim):
    def __init__(self, parameters, learning_rate = 1e-3, beta1 = .9, beta2 = .999, epsilon = 1e-8, weight_decay = 0):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        for weights, biases in zip(self.weights, self.biases):
           weights.m = np.zeros(weights.shape)
           biases.m = np.zeros(biases.shape)
           weights.v = np.zeros(weights.shape)
           biases.v = np.zeros(biases.shape)

    def step(self):
       self.t += 1
       for weights, biases in zip(self.weights, self.biases):
           weights.m = self.beta1 * weights.m + (1 - self.beta1) * weights.gradients
           weights.v = self.beta2 * weights.v + (1 - self.beta2) * weights.gradients * weights.gradients
           weights.m_hat = weights.m / (1 - self.beta1 ** self.t)
           weights.v_hat = weights.v / (1 - self.beta2 ** self.t)

           weights -= self.learning_rate * weights.m_hat / (np.sqrt(weights.v_hat) + self.epsilon) - self.weight_decay * weights

           biases.m = self.beta1 * biases.m + (1 - self.beta1) * biases.gradients
           biases.v = self.beta2 * biases.v + (1 - self.beta2) * biases.gradients * biases.gradients
           biases.m_hat = biases.m / (1 - self.beta1 ** self.t)
           biases.v_hat = biases.v / (1 - self.beta2 ** self.t)
           biases -= self.learning_rate * biases.m_hat / (np.sqrt(biases.v_hat) + self.epsilon)
