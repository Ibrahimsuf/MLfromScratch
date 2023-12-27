import unittest
from Layers import Linear, ConvolutionalLayer, MaxPooling
from dataloader import DataLoader
import numpy as np
from tensor import Tensor



class TestLayers(unittest.TestCase):
    
    def setUp(self):
        loader = DataLoader()
        self.X, self.train_labels = loader.load_data()

    
    def test_load_data(self):
        self.assertEqual(self.X.shape, (42000, 28, 28))
        self.assertEqual(self.train_labels.shape, (42000,))


    def test_add_padding(self):
        image = self.X[0].reshape(1, 28, 28)
        layer = ConvolutionalLayer(3, 5, 3, 2, 1)
        padded_image = layer.add_padding(image)

        self.assertEqual(padded_image.shape, (1, 30, 30))

    def test_convolve_subsection(self):
        subsection = np.array([[[1, 2, 3], [0, 0, 0], [0, 0, 0]]]).reshape((1, 1, 3, 3)).view(Tensor)
        layer = ConvolutionalLayer(1, 2, 3, 1, 1)

        # Horizontal edge detector (Sobel filter)
        horizontal_edge_detector = np.array([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]])

        # Vertical edge detector (Sobel filter)
        vertical_edge_detector = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]])
        # Combine the filters into a single array
        layer.filters = np.stack([horizontal_edge_detector, vertical_edge_detector])
        layer.filters = layer.filters[:, np.newaxis, :, :].view(Tensor)
        layer.bias = np.zeros_like(layer.bias).view(Tensor)
        
        output = layer.convolve_subsection(subsection)
        expected_output = np.array([-8, 2]).view(Tensor)


        # print(f"Output: {output}")
        # print(f"Expected Output: {expected_output}")

        self.assertTrue(np.array_equal(output, expected_output))
        
        output.backward()
        expected_gradient_filters = np.array([[[[1, 2, 3], [0, 0, 0], [0, 0, 0]]], [[[1, 2, 3], [0, 0, 0], [0, 0, 0]]]])
        expected_gradient_bias = np.array([1, 1])

        self.assertTrue(np.array_equal(layer.filters.gradients, expected_gradient_filters))
        self.assertTrue(np.array_equal(layer.bias.gradients, expected_gradient_bias))

    def test_max_pool(self):
        #testing with 1 filter
        image = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 ,11, 12], [13, 14, 15, 16]]]).reshape(1, 4, 4).view(Tensor)
        layer = MaxPooling(2, 2)
        output = layer(image)

        expected_output = np.array([[[6, 8], [14, 16]]]).view(Tensor)
        # print(f"Output: {output}")
        self.assertTrue(np.array_equal(output, expected_output))

        output.gradients = np.array([[[1, 2], [3, 4]]]).view(Tensor)
        output._backward()
        expected_gradient = np.array([[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]).view(Tensor)

        # print(f"Image gradients: {image.gradients}")
        self.assertTrue(np.array_equal(image.gradients, expected_gradient))


        #testing with 2 filters
        image = np.array([[[1 ,2], [3, 4]], [[5, 6], [7, 8]]]).reshape(2, 2, 2).view(Tensor)
        layer = MaxPooling(2, 2)
        output = layer(image)

        expected_output = np.array([[[4, 8]]]).view(Tensor)

        output.gradients = np.array([[[1]], [[2]]]).view(Tensor)
        output._backward()
        expected_gradient = np.array([[[0, 0], [0, 1]], [[0, 0], [0, 2]]]).view(Tensor)

        # print(f"Image gradients: {image.gradients}")

        self.assertTrue(np.array_equal(image.gradients, expected_gradient))

if __name__ == '__main__':
    unittest.main()
        
