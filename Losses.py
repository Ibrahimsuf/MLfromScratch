import numpy as np

class Cateogorical_cross_entropy:
    def __init__(self) -> None:
        pass
    def __call__(self, output, target):
        pass
    
    def forward(self, y_preds, y_trues):
        loss = 0 
        for y_pred in y_preds:
            for y_true in y_trues:
                loss += np.sum(y_true * np.log(y_pred + 1e-8))
    
    def backward(self):
        return -self.target / self.output
    