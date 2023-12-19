import os
import pickle
import numpy as np
class DataLoader:
    def load_data(self):
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
