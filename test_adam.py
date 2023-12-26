from optim import SGD, Adam
import numpy as np
from tensor import Tensor

a = np.array([5.9]).view(Tensor)
b = np.array([10.0]).view(Tensor)

loss = -a * a + np.array([2]).view(Tensor) * a

optimizer = Adam([[a], [b]])

for i in range(100):
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
