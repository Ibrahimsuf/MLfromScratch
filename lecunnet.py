from Layers import ConvolutionalLayer, Dense
import numpy as np

class Lecunnet():
    def __init__(self, input_shape = 28):
        self.input_shape = input_shape
        
        self.conv1 = ConvolutionalLayer(1, 8, 5,  2, 0)
        self.conv2 = ConvolutionalLayer(8, 10, 5, 2, 0)
        self.dense1 = Dense(10*7*7, 30)
        self.dense2 = Dense(30, 10, "softmax")
        self.num_classes = 10
    
    def __call__(self, image):

        if image.shape == (self.input_shape, self.input_shape):
            image = image[np.newaxis, :, :]

        image = self.normalize_input(image)
        # print("Image: ", image)
        output = self.conv1(image)
        # print("conv1 output: ", output)
        output = self.conv2(output)
        # print("conv2 output: ", output)
        output = output.reshape(-1)
        output = self.dense1(output)
        # print("dense1 output: ", output)
        output = self.dense2(output)
        # print("dense2 output: ", output)
        return output
    
    def normalize_input(self, image):
        return image/255.0 - 0.5
    

    # def cateogorical_cross_entropy(self, output, target):
    #     return -np.sum(target * np.log(output + 1e-8))
    
    def one_hot_encode(self, label):
        encoded = np.zeros(self.num_classes)
        encoded[label] = 1
        return encoded
    
    def find_loss(self, X, y):
        loss = 0
        for image, label in zip(X, y):
            output = self(image)
            # print(f"Output shape: {output.shape}")
            # print(f"Label shape: {self.one_hot_encode(int(label)).shape}")
            loss += self.cateogorical_cross_entropy(output, self.one_hot_encode(int(label)))
        return loss
    

