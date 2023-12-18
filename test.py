from tensor import Tensor
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = a.view(Tensor)
a.gradients = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])

b = a[:, 1:2]
print(b)