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
        # # Add new axis to the front of the image to represent the out channels
        # padded_image = padded_image[np.newaxis, :, :, :].view(Tensor)
        # output = np.zeros((self.out_channels, int(padded_image.shape[2]/self.stride), int(padded_image.shape[3]/self.stride))).view(Tensor)
        # # print(f"Output Shape: {output.shape}")
        # # print(f"End of range: {padded_image.shape[1] - self.size}")

        # for i in range(0, padded_image.shape[2] - self.size, self.stride):
        #     for j in range(0, padded_image.shape[3] - self.size, self.stride):
        #         # print(f"i: {i}, j: {j}")
        #         # print(f"i+size: {i+self.size}, j+size: {j+self.size}")
        #         image_section = padded_image[:, :, i:i+self.size, j:j+self.size]

        #         # print(f"Image Section: {image_section}")
        #         output[:, int(i/self.stride), int(j/self.stride)] = self.convolve_subsection(image_section)
        # return output

        filters_flattened = self.filters.reshape(self.out_channels, self.in_channels * self.size * self.size)
        patches = self.img2col(padded_image)
        output = filters_flattened @ patches + self.bias
        print(f"Output shape: {output.shape}")
        output = output.reshape(self.out_channels, int(padded_image.shape[1] - self.size/self.stride) + 1, int(padded_image.shape[2] - self.size/self.stride) + 1).view(Tensor)

        output.children.add(self.filters)
        output.children.add(image)

        return output
        def _backward():
            patches.gradients += output.gradients @ filters_flattened.T
            filters_flattened.gradients += patches.T @ output.gradients
            self.bias.gradients += output.gradients.sum(axis=0)

            filter_gradients = filters_flattened.gradients.reshape(self.filters.shape)
            image.gradients 



    def img2col(self, image):
        patch_size = self.in_channels * self.size * self.size
        num_locations_x = int((image.shape[1] - self.size)/self.stride + 1)
        num_patches = num_locations_x * num_locations_x
        patches = np.zeros((patch_size, num_patches))
        patch_number = 0
        for i in range(0, image.shape[1] - self.size + 1, self.stride):
            for j in range(0, image.shape[2] - self.size + 1, self.stride):
                patches[:, patch_number] = image[:, i:i + self.size, j:j + self.size].reshape(-1)
                patch_number += 1

        return patches.view(Tensor)
    
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
        assert input.shape[1] == self.in_features, f"Input shape must match in_features {input.shape[1]} != {self.in_features}"
        
        # print(f"Weights: {self.weights.shape}")
        # print(f"Input: {input.shape}")
        # print(f"Bias: {self.bias.shape}")
        # print(f"self.weights @ input: {(self.weights @ input).shape}")

        # print(f"Product.shpae: {(self.weights @ input).shape}")
        # print(f"Bias.shape: {self.bias.shape}")
        # print(f"Sum.shape: {(self.weights @ input + self.bias).shape}")
        return input @ self.weights + self.bias



class MaxPooling():
    def __init__(self, size, stride) -> None:
        self.size = size
        self.stride = stride

    def __call__(self, image):
        output = np.zeros((image.shape[0], int(image.shape[1]/self.stride), int(image.shape[2]/self.stride)))
        self.max_indices = np.zeros((image.shape[0], int(image.shape[1]/self.stride), int(image.shape[2]/self.stride), 2))
        for i in range(0, image.shape[1] - self.size + 1, self.stride):
            for j in range(0, image.shape[2] - self.size + 1, self.stride):
                image_section = image[:, i:i+self.size, j:j+self.size]

                max_idx = image_section.reshape(image_section.shape[0],-1).argmax(1)
                maxpos_vect = np.column_stack(np.unravel_index(max_idx, image_section[0,:,:].shape))
                maxpos_vect += np.array([i, j])
                # print(f"Max Pos Vect: {maxpos_vect}")
                # print(f"Max_idx: {max_idx}")

                self.max_indices[:, int(i/self.stride), int(j/self.stride), :] = maxpos_vect
                output[:, int(i/self.stride), int(j/self.stride)] = image_section.max(axis=(1,2))

        output = output.view(Tensor)

        def _backward():
            # print(f"Output gradients: {output.gradients.shape}")
            # print(f"Max indices: {self.max_indices.shape}")
            for i in range(0, self.max_indices.shape[1]):
                for j in range(0, self.max_indices.shape[2]):
                    for k in range(0, self.max_indices.shape[0]):
                        max_index_x, max_index_y = self.max_indices[k, i, j, :]
                        image.gradients[k, int(max_index_x), int(max_index_y)] += output.gradients[k, i, j]
        output.children.add(image)
        output._backward = _backward
        return output
                
