import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from Layers import ConvolutionalLayer, Dense
from lecunnet import Lecunnet


def main():
    X, train_labels = load_data()
    train_labels = train_labels.astype(int)
    # print(train_labels[:10])
    # plot_first_ten_images(X, train_labels)
    # test_convolve_image(X[0])
    # test_lecunnet(X[0])
    test_lecunnet(X[1], train_labels[1])
   



def test_lecunnet(image, y_label):
    lecunnet = Lecunnet()
    output = lecunnet(image)
    print(f"Output: {output}")
    print(f"Output Shape: {output.shape}")

    loss = output.cross_entropy(lecunnet.one_hot_encode(y_label))
    print(f"Loss: {loss}")
    loss.backward()

    # print(f"Dense 2 Gradients: {lecunnet.dense2.weights.gradients}")

def plot_first_ten_images(X, train_labels):
    # Plot the first ten images in the dataset
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i], cmap='gray')
        ax.set_title(f'Label: {train_labels[i]}')
    plt.tight_layout()
    plt.show()

def load_data():
    if os.path.exists('train_data.pkl'):
        with open('train_data.pkl', 'rb') as f:
            train = pickle.load(f)
    else:
        train = np.genfromtxt('train.csv', delimiter=',')
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train, f)

    # Remove the header row
    train = train[1:, :]
    # Split the data into features and labels
    train_features = train[:, 1:]
    train_labels = train[:, 0]
    X = train_features.reshape(train_features.shape[0], 28, 28)
    return X, train_labels

def test_add_padding(image):
    layer = ConvolutionalLayer(3, 1, 1, 1)

    padded_image = layer.add_padding(image)
    plot_image(image, padded_image)
    print(f'Padded Image Shape: {padded_image.shape}')
    print(f"Image Shape: {image.shape}")

def plot_image(image, image2, title1, title2):
    #plot padded image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(title1)
    axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2)
    plt.tight_layout()
    plt.show()

def test_convolve_subsection(image):
    subsection = image[15:18, 15:18].reshape(1, 3, 3)
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
    layer.filters = layer.filters[:, np.newaxis, :, :]
    # layer.filters = layer.filters.reshape(2, 1, 3, 3)

    layer.bias = np.zeros_like(layer.bias)
    
    output = layer.convolve_subsection(subsection)
    # print(f'Output Shape: {output.shape}')
    # print(f'Output: {output}')
    # print(f"Subsection: {subsection}")
    # print(f"Filters: {layer.filters[0, 0, :, :]}")

    # print(f'Filter Shape: {layer.filters.shape}')
    # print(f"Subsection Shape: {subsection.shape}")

    plot_image(subsection, layer.filters[0, :, :, :], 'Subsection', 'Filter')

    # print(f'Output Shape: {output.shape}')
    # print(f'Output: {output}')

def test_convolve_image(image):
    subsection = image.reshape(1, 28, 28)
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
    layer.filters = layer.filters[:, np.newaxis, :, :]
    # layer.filters = layer.filters.reshape(2, 1, 3, 3)

    layer.bias = np.zeros_like(layer.bias)
    
    output = layer(subsection)
    print(output.shape) 
    plot_image(subsection[0, :, :], output[0, :, :], 'Subsection', 'Convolved Image(Horizontal Edge Detector))')
    plot_image(subsection[0, :, :], output[1, :, :], 'Subsection', 'Convolved Image(Vertical Edge Detector))')

def test_find_loss():
    lecunnet = Lecunnet()
    X, train_labels = load_data()
    loss = lecunnet.find_loss(X[:100], train_labels[:100])
    print(f"Loss: {loss}")



def test_add_padding():
    layer = ConvolutionalLayer(1, 3, 1, 1, 1)
    image = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    padded_image = layer.add_padding(image)
    plot_image(image, padded_image, 'Image', 'Padded Image')
    print(f'Padded Image Shape: {padded_image.shape}')
    print(f"Image Shape: {image.shape}")

    
def test_dense():
    dense = Dense(3, 2)
    input = np.array([1, 2, 3])
    dense.weights = np.array([[1, 2, 3], [4, 5, 6]])
    dense.bias = np.array([1, 2])
    output = dense(input)
    print(f"Output: {output}")
    print(f"Output Shape: {output.shape}")


if __name__ == "__main__":
    main()
