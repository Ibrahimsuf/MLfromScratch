import numpy as np
from tensor import Tensor

class ConvolutionalLayer():
    def __init__(self, in_channels, out_chanels, size, stride, padding) -> None:
        self.in_channels = in_channels
        self.out_channels = out_chanels
        self.size = size
        self.stride = stride
        self.padding = padding

        self.filters = np.random.randn(self.out_channels, self.in_channels, size, size).view(Tensor)
        self.bias = np.random.randn(self.out_channels).view(Tensor)
    
    def __call__(self, image):
        padded_image = self.add_padding(image)
        # Add new axis to the front of the image to represent the out channels
        padded_image = padded_image[np.newaxis, :, :, :].view(Tensor)
        output = np.zeros((self.out_channels, int(padded_image.shape[2]/self.stride), int(padded_image.shape[3]/self.stride))).view(Tensor)
        # print(f"Output Shape: {output.shape}")
        # print(f"End of range: {padded_image.shape[1] - self.size}")

        for i in range(0, padded_image.shape[2] - self.size, self.stride):
            for j in range(0, padded_image.shape[3] - self.size, self.stride):
                # print(f"i: {i}, j: {j}")
                # print(f"i+size: {i+self.size}, j+size: {j+self.size}")
                image_section = padded_image[:, :, i:i+self.size, j:j+self.size]

                # print(f"Image Section: {image_section}")
                output[:, int(i/self.stride), int(j/self.stride)] = self.convolve_subsection(image_section)
        return output



    def convolve_subsection(self, subsection):
        assert subsection.shape == (1, self.in_channels, self.size, self.size), f"Subsection shape must match filter shape {subsection.shape} != {(1, self.in_channels, self.size, self.size)}"

        #self.filters.shape = (out_channels, in_channels, size, size)
        # For each out channel we want to take the element wise product of the filter and the image (this includes going across multiple channels)
        #then we want to sum accross the all the columns but the channel column
        # product = self.filters * subsection
        # sum = (product).sum(axis=(1,2, 3))
        # out = (product).sum(axis=(1,2, 3)) + self.bias

        # print(f"Product: {product}")
        # print(f"Product Shape: {product.shape}")

        # print(f"Sum: {sum}")
        # print(f"Sum Shape: {sum.shape}")
        # print(f"OUT: {out}")
        # print(f"OUT SHAPE: {out.shape}")
        
        #We keep dims and then reshape instead of keepdims = False because it works better for computing the gradients in 2 steps.
        #This is probaby suboptimal and should be changed
        return (self.filters * subsection).sum(axis=(1,2,3), keepdims = True).reshape(self.out_channels) + self.bias
        
    def add_padding(self, image):
        if self.padding == 0:
            return image
        

        padded_image = np.zeros((image.shape[0], image.shape[1] + 2*self.padding, image.shape[2] + 2*self.padding))
        padded_image[:, self.padding:-self.padding, self.padding:-self.padding] = image
        return padded_image

class Linear():
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(self.in_features, self.out_features).view(Tensor) / np.sqrt(self.in_features)
        self.bias = np.zeros(self.out_features).view(Tensor)

    def __call__(self, input):
        assert input.shape[1] == self.in_features, f"Input shape must match in_features {input.shape} != {(self.in_features, 1)}"
        
        # print(f"Weights: {self.weights.shape}")
        # print(f"Input: {input.shape}")
        # print(f"Bias: {self.bias.shape}")
        # print(f"self.weights @ input: {(self.weights @ input).shape}")

        # print(f"Product.shpae: {(self.weights @ input).shape}")
        # print(f"Bias.shape: {self.bias.shape}")
        # print(f"Sum.shape: {(self.weights @ input + self.bias).shape}")
        return input @ self.weights + self.bias