import numpy as np
from itertools import zip_longest

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
            self.children = set()
        elif isinstance(obj, np.ndarray):
            self.gradients = np.zeros(self.shape)
            self._backward = lambda: None
            self.children = set()
    
    def __repr__(self) -> str:
        return "Values " + super().__repr__() + '\n' + 'Gradients: ' + str(self.gradients)
    
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
            self.gradients[index] = value.gradients if value.gradients is not None else np.zeros(value.shape)
        super().__setitem__(index, value)

    
    def __add__(self, other):
        # We may be adding a (32, 10) tensor to a (10, ) tensor so the 10 broadcasts 32 times
        # We need to make sure that the gradients are added correctly
        out = super().__add__(other)
        out.children.add(self)
        out.children.add(other)

        def _backward():
                self_missing, other_missing = Tensor.get_different_dimensions(self.gradients, other.gradients)
                
                # print(f"Self gradients: {self.gradients.shape}")
                # print(f"Other gradients: {other.gradients.shape}")

                # print(f"Self missing: {self_missing}")
                # print(f"Other missing: {other_missing}")
                
                self.gradients += out.gradients.sum(axis=self_missing, keepdims=False)
                other.gradients += out.gradients.sum(axis=other_missing, keepdims=False)

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

            self.gradients += (out.gradients.T * np.ones_like(self).view(np.ndarray).T).T
        
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = super().reshape(*shape)
        out.children.add(self)


        def _backward():
            self.gradients += out.gradients.reshape(self.shape)
            # print(f"Self gradients: {self.gradients}")
        
        out._backward = _backward
        return out

    def backward(self):
        # print(f'In backward with self: {self}')
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
        # print(f"Topo: {topo}")
        for v in reversed(topo):
            v._backward()
            # print(v)

    def __matmul__(self, other):
        out = super().__matmul__(other)
        out.children.add(self)
        out.children.add(other)

        def _backward():
            self.gradients += out.gradients @ other.T.view(np.ndarray)
            other.gradients += self.T.view(np.ndarray) @ out.gradients
        out._backward = _backward
        return out

    def __hash__(self) -> int:
        return id(self)

    def cross_entropy(self, target):
        """Returns the cross entropy loss between the target and the softmax of this tensor"""
        logits = self.view(np.ndarray) - np.max(self.view(np.ndarray), axis=1, keepdims=True)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        out =  (-np.sum(target * np.log(probs + 1e-8)) / target.shape[0])
        out = out.reshape(1).view(Tensor)
        out.children.add(self)
        def _backward():
            self.gradients += out.gradients * (probs - target) / probs.shape[0]

        out._backward = _backward
        return out

    def convolve(self, other, stride=1):
        pass
        

    @staticmethod
    def get_different_dimensions(arr1, arr2):
        axis = max(arr1.ndim, arr2.ndim) - 1
        arr_1_missing = []
        arr_2_missing = []
        for arr1_dim, arr2_dim in zip_longest(arr1.shape[::-1], arr2.shape[::-1]):
            if arr1_dim != arr2_dim:
                if not arr1_dim or arr1_dim == 1:
                    arr_1_missing.append(axis)
                if not arr2_dim or arr2_dim == 1:
                    arr_2_missing.append(axis)
            axis -= 1
        
        # print(f"Arr 1 missing: {arr_1_missing}")
        # print(f"Arr 2 missing: {arr_2_missing}")

        return tuple(arr_1_missing), tuple(arr_2_missing)

    @staticmethod
    def image2col(image, stride, filter_size):
        """Converts an image to a column matrix"""
        

