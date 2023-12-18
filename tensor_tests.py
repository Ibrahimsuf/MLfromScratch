import sys
import unittest
from tensor import Tensor
import numpy as np

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

    
    def test_sigmoid(self):
        a = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9]])
        a = a.view(Tensor)

        b = a.sigmoid()
        b.gradients = np.array([[5, 2, 7], [2, 0, 6], [8, 6, 6]])

        b._backward()
        self.assertTrue(np.allclose(b.view(np.ndarray), np.array([[0.73105858, 0.88079708, 0.95257413], [0.01798621, 0.99330715, 0.99752738], [0.99908895, 0.00033535, 0.99987661]]), atol=1e-8))
        self.assertTrue(np.allclose(a.gradients, b * (1 - b) * b.gradients, atol=1e-8))


if __name__ == '__main__':
    unittest.main()