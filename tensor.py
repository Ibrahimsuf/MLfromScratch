import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # print('In __new__ with class %s' % cls)
        return super().__new__(cls, *args, **kwargs)
    def __array_finalize__(self, obj):
        # print('In array_finalize:')
        # print('   self type is %s' % type(self))
        # print('   obj type is %s' % type(obj))

        if obj is None:
            self.gradients = np.zeros(self.shape)
            self._backward = lambda: None
        elif isinstance(obj, np.ndarray):
            self.gradients = np.zeros(self.shape)
            self._backward = lambda: None
            self.children = set()
    
    def __repr__(self) -> str:
        return super().__repr__() + '\n' + 'Gradients: ' + str(self.gradients)
    
    def __str__(self) -> str:
        return super().__str__() + '\n' + 'Gradients: ' + str(self.gradients)
    
    def __getitem__(self, index):
        # print("In __getitem__ with index %s" % index)
        # print("self: ", self)
        result = super().__getitem__(index)
        if isinstance(result, Tensor):
            result.gradients = self.gradients[index]
        return result

    def __setitem__(self, index, value):
        if isinstance(value, Tensor):
            self.gradients[index] = value.gradients if value.gradients else np.zeros(value.shape)
        super().__setitem__(index, value)

    
    def __add__(self, other):
        out = super().__add__(other)
        out.children.add(self)
        out.children.add(other)

        def _backward():
                self_missing, other_missing = Tensor.get_different_dimensions(self.gradients, other.gradients)
                self.gradients += out.gradients.sum(axis=self_missing, keepdims=True)
                other.gradients += out.gradients.sum(axis=other_missing, keepdims=True)

            # if self is broadcasted to be the same shape as other
            #if self.gradients has dims (1, 3) and out.gradients has dims (3, 3) we want to sum the gradients on axis 0


            # self.gradients += out.gradients
            # other.gradients += out.gradients
            
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = super().__mul__(other)
        out.children.add(self)
        out.children.add(other)

        def _backward():
            # print(f"Self: {self.shape}")
            # print(f"Self: {self}")
            # print(f"Other: {other.shape}")
            # print(f"Other: {other}")
            # print(f"Out: {out.shape}")
            self_missing, other_missing = Tensor.get_different_dimensions(self.gradients, other.gradients)
            self.gradients += (out.gradients * other.view(np.ndarray)).sum(axis=self_missing, keepdims=True)
            other.gradients += (out.gradients * self.view(np.ndarray)).sum(axis=other_missing, keepdims=True)
        
        out._backward = _backward
        return out
    
    def relu(self) -> 'Tensor':
        out = np.maximum(self, 0)
        out.children.add(self)
        def _backward():
            self.gradients += out.gradients * ((self > 0).view(np.ndarray).astype(self.dtype))
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> 'Tensor':
        out = 1 / (1 + np.exp(-self))
        out.children.add(self)
        def _backward():
            self.gradients += out.gradients * out * (1 - out)
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=True):
        out = super().sum(axis=axis, keepdims=keepdims)
        out.children.add(self)
        def _backward():
            # print(f"Ot gradients: {out.gradients}")
            # print(f"Self shape: {self.shape}")
            # print(f"Self gradients: {self.gradients}")
            
            #We need to take the transpose becuase numpy broadcasting starts from the last dimension and the channels out is the first dimension
            #https://stackoverflow.com/questions/22603375/numpy-broadcast-from-first-dimension

            # print(f"Out gradients: {out.gradients}")
            # print(f"Out gradients transpose: {out.gradients.T}")
            # print(f"Ones like self: {np.ones_like(self)}")
            # print(f"Ones like self transpose: {np.ones_like(self).T}")
            # print(f"Out gradients transpose * ones like self transpose: {(out.gradients.T * np.ones_like(self).T)}")

            self.gradients += (out.gradients.T * np.ones_like(self).T).T
        
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = super().reshape(*shape)
        out.children.add(self)
        def _backward():
            self.gradients += out.gradients.reshape(self.shape)
        
        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.gradients = np.ones(self.shape)
        for v in reversed(topo):
            v._backward()

    def __matmul__(self, other):
        out = super().__matmul__(other)
        out.children.add(self)
        out.children.add(other)

        def _backward():
            self.gradients += out.gradients @ other.T
            other.gradients += self.gradients.T @ out.gradients
        
        # we need to reshape these to be 2d arrays so the matrix multiplication works
        # notes
        out._backward = _backward
        return out

    def __hash__(self) -> int:
        return id(self)


    def softmax(self):
        self_values = self.view(np.ndarray)
        self_values = self_values - np.max(self_values)
        # print(f"Self values: {self_values}")
        out = (np.exp(self_values) / np.sum(np.exp(self_values))).view(Tensor)
        out.children.add(self)
        def _backward():
            out_ndarray = out.view(np.ndarray)
            self.gradients += out.gradients @ (np.diag(out_ndarray) - np.outer(out_ndarray, out_ndarray))
        
        out._backward = _backward

        # print(f"Out: {out}")
        return out

    def cross_entropy(self, target):
        out = -np.log(self[np.where(target == 1)] + 1e-8)
        out.children.add(self)

        def _backward():
            self.gradients += out.gradients * (-target.view(np.ndarray) / (self.view(np.ndarray) + 1e-8))

        out._backward = _backward
        return out


    @staticmethod
    def get_different_dimensions(arr1, arr2):
        # Ensure both arrays are numpy arrays
        arr1, arr2 = np.asarray(arr1), np.asarray(arr2)

        # Get the shapes of the arrays
        shape1, shape2 = arr1.shape, arr2.shape

        # Find the minimum length of the shapes
        min_len = min(len(shape1), len(shape2))

        # Compare the dimensions of the arrays up to the minimum length
        arr1_missing = [i for i in range(min_len) if shape1[i] < shape2[i]]
        arr2_missing = [i for i in range(min_len) if shape2[i] < shape1[i]]
        return tuple(arr1_missing), tuple(arr2_missing)


