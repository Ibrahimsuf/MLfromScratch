import torch 
import torch.nn as nn
import pickle
import numpy as np
import unittest
import tqdm as tqdm
from torch_nn import NeuralNetwork
from numpyMLP import NeuralNetworkNumpy
from tensor import Tensor
torch.manual_seed(0)



class Test_Loss(unittest.TestCase):
    def setUp(self) -> None:
        with open("train_data.pkl", "rb") as f:
            self.train = pickle.load(f)
        
        self.train = self.train[1:, :]
        self.train_features = self.train[:, 1:].reshape(self.train.shape[0], 28, 28)
        self.train_labels = self.train[:, 0]
        self.X_numpy = self.train_features

        self.X_torch = torch.from_numpy(self.X_numpy).float()

        learning_rate = 1e-3
        batch_size = 32
        epochs = 10

        self.torch_nn = NeuralNetwork()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.torch_nn.parameters(), lr=learning_rate)

        Test_Loss.train_loop(self.torch_nn, loss_fn, optimizer, batch_size, self.X_torch, self.train_labels)

        self.numpy_nn = NeuralNetworkNumpy()
        self.numpy_nn.layer1.weights = self.torch_nn.linear_relu_stack[0].weight.detach().numpy().T.view(Tensor)
        self.numpy_nn.layer1.bias = self.torch_nn.linear_relu_stack[0].bias.detach().numpy().view(Tensor)
        self.numpy_nn.layer2.weights = self.torch_nn.linear_relu_stack[2].weight.detach().numpy().T.view(Tensor)
        self.numpy_nn.layer2.bias = self.torch_nn.linear_relu_stack[2].bias.detach().numpy().view(Tensor)
        self.numpy_nn.layer3.weights = self.torch_nn.linear_relu_stack[4].weight.detach().numpy().T.view(Tensor)
        self.numpy_nn.layer3.bias = self.torch_nn.linear_relu_stack[4].bias.detach().numpy().view(Tensor)


        self.batch_torch = self.X_torch[0:32,:,:]
        self.batch_label_torch = torch.from_numpy(Test_Loss.one_hot_encode(self.train_labels[0:32]))
        self.batch_numpy = self.X_numpy[0:32,:,:]
        self.batch_label_numpy = Test_Loss.one_hot_encode(self.train_labels[0:32])
    

    
    def test_loss(self):
        torch_pred = self.torch_nn(self.batch_torch)
        numpy_pred = self.numpy_nn.forward(self.batch_numpy)
        # print(f"torch_pred: {torch_pred[0]}")
        # print(f"numpy_pred: {numpy_pred[0]}")
        self.assertTrue(np.allclose(torch_pred.detach().numpy(), numpy_pred, atol=1e-5), "Torch and Numpy predictions are not close enough")

        torch_loss = nn.CrossEntropyLoss()(torch_pred, self.batch_label_torch)
        numpy_loss = numpy_pred.cross_entropy(self.batch_label_numpy)

        # print(f"torch_loss: {torch_loss}")
        # print(f"numpy_loss: {numpy_loss}")
        self.assertTrue(np.allclose(torch_loss.detach().numpy(), numpy_loss, atol=1e-5), "Torch and Numpy losses are not close enough")
        
        numpy_loss.backward()
        torch_loss.backward()

        # print(f"torch_nn.layer1.weights.grad: {self.torch_nn.linear_relu_stack[0].weight.grad.T[:,30:50]}")
        # print(f"numpy_nn.layer1.weights.grad: {self.numpy_nn.layer1.weights.gradients[:,30:50]}")

        self.assertTrue(np.allclose(self.torch_nn.linear_relu_stack[0].weight.grad.T.detach().numpy(), self.numpy_nn.layer1.weights.gradients, atol=1e-5), "Torch and Numpy gradients are not close enough")

        
    

    @staticmethod
    def one_hot_encode(labels):
        encoded = np.zeros((labels.shape[0], 10))
        encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1
        return encoded

    @staticmethod
    def train_loop(model, loss_fn, optimizer, batch_size, X, train_labels):
        model.train()
        epochs = 10
        num_elements = X.shape[0]
        num_train_batches = epochs * (num_elements // batch_size)
        for i in tqdm.tqdm(range(num_train_batches)):
            # Compute prediction error
            batch_size = 32
            indices = np.random.choice(num_elements, batch_size, replace=False)

            batch = X[indices,:,:]
            y_one_hot = Test_Loss.one_hot_encode(train_labels[indices])
            batch_labels = torch.from_numpy(y_one_hot).float()


            pred = model(batch)
            loss = loss_fn(pred, batch_labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}")


if __name__ == "__main__":
    unittest.main()