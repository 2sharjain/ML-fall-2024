import pandas as pd
import numpy as np
import sys
class Logistic_Regression:
    def __init__(self):
        self.lr = 0.01
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1
        self.variance = 1


    

    def train_MAP(self, x, y):
        num_samples, dim = x.shape
        w = np.zeros((1, dim))
        idx = np.arange(num_samples)

        for epoch in range(self.epoch):
            np.random.shuffle(idx)
            x, y = x[idx], y[idx]

            for i in range(num_samples):
                x_i = x[i:i + 1]  # Efficient slicing to retain 2D shape
                y_i = y[i]

                # Compute gradient
                tmp = y_i * np.dot(w, x_i.T).item()
                gradient = -num_samples * y_i * x_i / (1 + np.exp(tmp)) + w / self.variance

                # Update weights
                learning_rate = self.lr / (1 + (self.lr / self.d) * epoch)
                w -= learning_rate * gradient

        return w.T  # Return w as a column vector

    def train_MLE(self, x, y):
        num_samples, dim = x.shape
        w = np.zeros((1, dim))
        idx = np.arange(num_samples)

        for epoch in range(self.epoch):
            np.random.shuffle(idx)
            x, y = x[idx], y[idx]

            for i in range(num_samples):
                x_i = x[i:i + 1]  # Efficient slicing to maintain 2D shape
                y_i = y[i]

                # Compute gradient
                tmp = y_i * np.dot(w, x_i.T).item()
                gradient = -num_samples * y_i * x_i / (1 + np.exp(tmp))

                # Update weights
                learning_rate = self.lr / (1 + (self.lr / self.d) * epoch)
                w -= learning_rate * gradient

        return w.T  # Return w as a column vector



train_data = pd.read_csv('bank-note/train.csv', header=None)

# Process data
raw = train_data.values
num_rows, num_cols = raw.shape

train_x = np.copy(raw)
train_x[:, -1] = 1 
train_y = 2 * raw[:, -1] - 1


# print(train_x)
# print(train_y)

test_data = pd.read_csv('bank-note/test.csv', header=None)
raw = test_data.values
numrows, num_col = raw.shape
test_x = np.copy(raw)
test_x[:, -1] = 1 
test_y = 2 * raw[:, -1] - 1

gamma_set = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
model= Logistic_Regression()
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

for variance in variances:
    model.variance = variance
    print('Variance:', variance)
    if sys.argv[1] == "map": w= model.train_MAP(train_x, train_y)
    elif sys.argv[1] == "mle": w= model.train_MLE(train_x, train_y)

    pred = np.sign(np.matmul(train_x, w))
    pred[pred == 0] = -1

    train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
    print(f'Training error: {train_err:.4f},\t  Testing error: {test_err:.4f}\n\n')