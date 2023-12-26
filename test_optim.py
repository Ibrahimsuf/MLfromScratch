import pickle
import numpy as np
import unittest
import tqdm as tqdm
from numpyMLP import NeuralNetworkNumpy
from optim import SGD, Adam
from tensor import Tensor
import idx2numpy



class Test_Optim(unittest.TestCase):
    def setUp(self) -> None:
        with open("train_data.pkl", "rb") as f:
            self.train = pickle.load(f)
        
        self.train = self.train[1:, :]
        self.train_features = self.train[:, 1:].reshape(self.train.shape[0], 28, 28)
        self.train_labels = self.train[:, 0]
        self.X_numpy = self.train_features
        self.test_labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
        self. test_images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')


    def test_training(self):
        learning_rate = 1e-3
        batch_size = 32
        epochs = 10

        numpy_nn = NeuralNetworkNumpy()
        optimizer = SGD(numpy_nn.parameters(), learning_rate)

        Test_Optim.train_loop(numpy_nn, optimizer, batch_size, self.X_numpy, self.train_labels, epochs)

        test_acc = Test_Optim._test_loop_numpy(numpy_nn, self.test_images, self.test_labels)
        print(f"Test Loss: {test_acc}")
        train_acc = Test_Optim._test_loop_numpy(numpy_nn, self.X_numpy, self.train_labels)
        print(f"Train Loss: {test_acc}")

        self.assertTrue(test_acc > 0.9, "Test accuracy is too low")
        self.assertTrue(train_acc > 0.9, "Train accuracy is too low")

    def test_adam(self):
        batch_size = 32
        epochs = 5
        numpy_nn = NeuralNetworkNumpy()
        optimizer = Adam(numpy_nn.parameters())

        Test_Optim.train_loop(numpy_nn, optimizer, batch_size, self.X_numpy, self.train_labels, epochs)
        
        test_acc = Test_Optim._test_loop_numpy(numpy_nn, self.test_images, self.test_labels)
        print(f"Test Loss: {test_acc}")
        train_acc = Test_Optim._test_loop_numpy(numpy_nn, self.X_numpy, self.train_labels)
        print(f"Train Loss: {test_acc}")

        self.assertTrue(test_acc > 0.9, "Test accuracy is too low")
        self.assertTrue(train_acc > 0.9, "Train accuracy is too low")

    @staticmethod
    def one_hot_encode(labels):
        encoded = np.zeros((labels.shape[0], 10))
        encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1
        return encoded

    @staticmethod
    def train_loop(model, optimizer, batch_size, X, train_labels, epochs = 10):
        num_elements = X.shape[0]
        num_train_batches = epochs * (num_elements // batch_size)
        for i in tqdm.tqdm(range(num_train_batches)):
            # Compute prediction error
            batch_size = 32
            indices = np.random.choice(num_elements, batch_size, replace=False)

            batch = X[indices,:,:]
            batch_labels = Test_Optim.one_hot_encode(train_labels[indices])

            pred = model.forward(batch)
            loss = pred.cross_entropy(batch_labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}")

    @staticmethod
    def _test_loop_numpy(model, X, test_labels, batch_size = 32):
    # model.eval()
        num_test_samples = X.shape[0]
        num_test_batches = num_test_samples // batch_size
        test_acc = 0
        for i in tqdm.tqdm(range(num_test_batches)):
            batch = X[i*batch_size:(i+1)*batch_size,:,:]
            batch_labels = Test_Optim.one_hot_encode(test_labels[i*batch_size:(i+1)*batch_size, ])

            pred = model.forward(batch)
            # loss = loss_fn(pred, batch_labels)
            test_acc += (pred.argmax(1) == batch_labels.argmax(1)).sum().item()
            # incorrect += (pred.argmax(1) != batch_labels.argmax(1)).type(torch.float).sum().item()

        test_acc /= num_test_samples
        # print(f"Test Error: \n Accuracy: {(100*test_acc):>0.1f}%")
        return test_acc
if __name__ == "__main__":
    unittest.main()