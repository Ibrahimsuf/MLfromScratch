import sys
import unittest
from tensor import Tensor
import numpy as np
import torch
import torch.nn.functional as F

class TestTensor(unittest.TestCase):
    def test_tensor_creation(self):
        t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t = t.view(Tensor)
        self.assertEqual(t[0,0], 1)
        self.assertEqual(t[1,0], 4)
        self.assertEqual(t[2,2], 9)

    def test_tensor_gradients(self):
        t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t = t.view(Tensor)
        t.gradients = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])
        self.assertEqual(t.gradients[0,0], 5)
        self.assertEqual(t.gradients[1,1], 2)
        self.assertEqual(t.gradients[2,1], 5)

    def test_tensor_splicing(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        a = a.view(Tensor)
        a.gradients = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])

        b = a[:, 1:2]

        self.assertEqual(b[0][0], 2)
        self.assertEqual(b[1][0], 5)
        self.assertEqual(b[2][0], 8)
        self.assertEqual(b.gradients[0][0], 6)
        self.assertEqual(b.gradients[1][0], 2)
        self.assertEqual(b.gradients[2][0], 5)

    # def test_get_single_element(self):
    #     print("Test runs ")
    #     a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #     a = a.view(Tensor)
    #     a.gradients = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])

    #     b = a[1, 1]

    #     self.assertEqual(b, 5)
    #     self.assertEqual(b.gradients, 2)
    def test_backprop_adding(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        a = a.view(Tensor)
        
        b = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])
        b = b.view(Tensor)

        c = a + b
        c.gradients = np.array([[5, 2, 7], [1, 2, 6], [8, 6, 6]])

        c._backward()

        self.assertTrue(np.array_equal(a.gradients, c.gradients))
        self.assertTrue(np.array_equal(b.gradients, c.gradients))
    
    def test_multiply(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        a = a.view(Tensor)
        
        b = np.array([[5, 6, 7], [1, 2, 3], [4, 5, 6]])
        b = b.view(Tensor)

        c = a * b
        c.gradients = np.array([[5, 2, 7], [1, 2, 6], [8, 6, 6]])

        c._backward()


        self.assertTrue(np.array_equal(a.gradients, b.view(np.ndarray) * c.gradients))
        self.assertTrue(np.array_equal(b.gradients, a.view(np.ndarray) * c.gradients))
    
    def test_relu(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)

        b = a.relu()
        b.gradients = np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])

        b._backward()

        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 2, 7], [0, 0, 6], [8, 0, 6]])))

    def test_broadcasting(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]]).view(Tensor)
        b = np.array([[1, 2, 3]]).view(Tensor) 

        c = a + b
        d = a * b

        c.gradients = np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])
        d.gradients = np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])

        c._backward()
        # d._backward()

        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])))
        self.assertTrue(np.array_equal(b.gradients, np.array([[15, 8, 19]]) ))


        a.gradients = np.zeros(a.shape)
        b.gradients = np.zeros(b.shape)

        d._backward()

        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 4, 21], [2, 0, 18], [8, 12, 18]])))
        self.assertTrue(np.array_equal(b.gradients, np.array([[53, -44, 111]])))

    def test_sigmoid(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)

        b = a.sigmoid()
        b.gradients = np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])

        b._backward()
        self.assertTrue(np.allclose(b.view(np.ndarray), np.array([[0.73105858, 0.88079708, 0.95257413], [0.01798621, 0.99330715, 0.99752738], [0.99908895, 0.00033535, 0.99987661]]), atol=1e-8))
        
        sigmoid_gradient = b.gradients * b.view(np.ndarray) * (1 - b.view(np.ndarray))
        self.assertTrue(np.allclose(a.gradients, sigmoid_gradient, atol=1e-8))

    def test_sum(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)

        b = a.sum(axis=0, keepdims=False)
        b.gradients = np.array([[5, 2, 7]])
        b._backward()

        # print(f"Gradients: {a.gradients}")
        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 2, 7], [5, 2, 7], [5, 2, 7]])))
    def test_sum_keep_dims(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)
        c = a.sum(axis=1, keepdims=True)
        c.gradients = np.array([[5], [2], [7]])
        c._backward()
        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 5, 5], [2, 2, 2], [7, 7, 7]])))

    def test_backward(self):
        # Create a tensor and set requires_grad=True to track computation with it
        x = torch.tensor([1.0, -1.0, 2.0, -2.0], requires_grad=True)
        y = torch.tensor([1.0, 1.0, 4.0, 1.0], requires_grad=True)
        # Apply ReLU operation
        relu_torch = F.relu(x + y)

        # Apply sigmoid operation
        sigmoid_torch = torch.sigmoid(relu_torch)

        # Apply sum operation
        s = torch.sum(sigmoid_torch, dim = 0, keepdim=True)

        # Use backward to compute gradients
        s.backward()

        a = np.array([1.0, -1.0, 2.0, -2.0]).view(Tensor)
        b= np.array([1.0, 1.0, 4.0, 1.0]).view(Tensor)

        relu = (a + b).relu()
        sigmoid_mine = relu.sigmoid()
        sum_mine = sigmoid_mine.sum()

        sum_mine.backward()

        self.assertTrue(np.allclose(a.gradients, x.grad, atol=1e-6))
        self.assertTrue(np.allclose(b.gradients, y.grad, atol=1e-6))

    def test_reshape(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)
        b = a.reshape(1, 9)
        self.assertTrue(np.array_equal(b.view(np.ndarray), np.array([[1, 2, 3, -4, 5, 6, 7, -8, 9]])))

        b.gradients = np.array([[5, 2, 7, 7, 1, 0, 0, 4, 2]])
        b._backward()
        self.assertTrue(np.array_equal(a.gradients, np.array([[5, 2, 7], [7, 1, 0], [0, 4, 2]])))

    def test_cross_entropy(self):
        a = np.array([-5, 3, 7, 8]).view(Tensor)
        y = np.array([0, 1, 0, 0])

        loss = a.cross_entropy(y)

        a_torch = torch.tensor([-5.0, 3.0, 7.0, 8.0], requires_grad=True)
        y_torch = torch.tensor([1])

        torch_loss = torch.nn.CrossEntropyLoss()
        loss_torch_value = torch_loss(a_torch.view(1, -1), y_torch)


        # print(f"My Loss: {loss}")
        # print(f"Torch Loss: {loss_torch_value}")
        self.assertTrue(np.allclose(loss.view(np.ndarray), loss_torch_value.detach().numpy(), atol=1e-6))
        
        loss.backward()
        loss_torch_value.backward()

        self.assertTrue(np.allclose(a.gradients, a_torch.grad, atol=1e-6))


        


if __name__ == '__main__':
    unittest.main()