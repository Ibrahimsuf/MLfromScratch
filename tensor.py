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

        def _backward():
            self.gradients += out.gradients
            other.gradients += out.gradients
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = super().__mul__(other)

        def _backward():
            self.gradients += out.gradients * other.view(np.ndarray)
            other.gradients += out.gradients * self.view(np.ndarray)
        
        out._backward = _backward
        return out
    
    def relu(self) -> 'Tensor':
        out = np.maximum(self, 0)
        def _backward():
            self.gradients += out.gradients * ((self > 0).view(np.ndarray).astype(self.dtype))
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> 'Tensor':
        out = 1 / (1 + np.exp(-self))
        def _backward():
            self.gradients += out.gradients * out * (1 - out)
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = super().sum(axis=axis, keepdims=keepdims)
        def _backward():
            self.gradients += out.gradients * np.ones_like(self)
        
        out._backward = _backward
        return out