from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

bank_train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

X_train = bank_train_df.iloc[:, :-1].values
Y_train = bank_train_df.iloc[:, -1].values


X_test = bank_test_df.iloc[:, :-1].values
Y_test = bank_test_df.iloc[:, -1].values

lr = 0.001
d = 0.01
T = 10
nodes_list = [5, 10, 25, 50, 100]
class mythreelayernetwork:

    def __init__(self, num_features, num_nodes, weights_, d, initial_learning_rate):
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.lr = initial_learning_rate
        self.d = d
        if weights_ == "random":
            self.weights = [np.random.randn(num_features + 1, num_nodes), np.random.randn(num_nodes + 1, num_nodes),
                            np.random.randn(num_nodes + 1, 1)]
        elif weights_ == "zeros":
            self.weights = [np.zeros((num_features + 1, num_nodes)), np.zeros((num_nodes + 1, num_nodes)),
                            np.zeros((num_nodes + 1, 1))]
        else:
            self.weights = weights

        self.lr_inc = self.lr_a

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def lr_a(self, epoch):
        return self.lr / (1 + (self.lr / self.d) * epoch)
    

    def _forward_pass(self, input_data: np.ndarray):
        activation_outputs = [input_data]
        z = []

        for weight in self.weights:
            input_x = np.hstack((activation_outputs[-1], np.ones(1)))  # Add bias
            z_curr = np.dot(input_x, weight)
            z.append(z_curr)
            activation_outputs.append(self.sigmoid(z_curr))
        
        return activation_outputs, z

    def _backward_pass(self, activation_outputs: list, z: list, target: np.ndarray):
        """Perform a backward pass to calculate deltas."""
        delta = [activation_outputs[-1] - target]
        
        for layer in range(len(self.weights) - 1, 0, -1):
            weight_no_bias = self.weights[layer][:-1, :]  # Exclude bias weights
            delta_prev = delta[-1] @ weight_no_bias.T * self.sigmoid_derivative(activation_outputs[layer])
            delta.append(delta_prev)
        
        delta.reverse()
        return delta

    def _update_weights(self, activation_outputs: list, delta: list, learning_rate: float):
        """Update the weights using backpropagation deltas."""
        for layer in range(len(self.weights)):
            input_x = np.hstack((activation_outputs[layer], np.ones(1)))  # Add bias
            self.weights[layer] -= learning_rate * np.outer(input_x, delta[layer])

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(x))
            train_x, train_y = x[indices], y[indices]
            
            for i in range(len(train_x)):
                # Forward pass
                activation_outputs, z = self._forward_pass(train_x[i])
                
                # Backward pass
                delta = self._backward_pass(activation_outputs, z, train_y[i])
                
                # Update weights
                self._update_weights(activation_outputs, delta, self.lr)

            # Log training progress
            training_error = self.evaluate(x, y)
            # print(f'Epoch {epoch + 1}/{epochs}, Training Error: {training_error}')
            # # loss
            # loss = 0
            # for i in range(len(x)):
            #     a = [x[i]]
            #     for j in range(len(self.weights)):
            #         input_x = np.hstack((a[j], np.ones(1)))
            #         a.append(self.sigmoid(np.dot(input_x, self.weights[j])))
            #     loss += (a[-1] - y[i]) ** 2
            # loss /= len(x)
            # print(f"Epoch: {epoch}, Loss: {loss}")

    def predict(self, x: np.ndarray) -> int:
        """Make a prediction for a single input sample."""
        for i in range(len(x)):
            a = [x[i]]  # input layer
            for j in range(len(self.weights)):

                input_x = np.hstack((a[j], np.ones(1)))
                a.append(self.sigmoid(np.dot(input_x, self.weights[j])))
        # print(a)        
        return 1 if a[-1] >= 0.5 else 0

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) != y)
    



def printerrors(nn):
    print(f'Training Error:\t {nn.evaluate(X_train, Y_train):.3f}')
    print(f'Testing Error: \t {nn.evaluate(X_test, Y_test):.3f}\n')
    return




print("------randomly initialized weights--------")
for num_nodes in nodes_list:
    print("Number of nodes: " + str(num_nodes))
    nn_model = mythreelayernetwork(X_train.shape[1], num_nodes, "random", d, lr)
    nn_model.train(X_train, Y_train, T)
    printerrors(nn_model)
    

# (2c)
print("------weights initialized to zero--------")
for num_nodes in nodes_list:
    print("Number of nodes: " + str(num_nodes))
    nn_model = mythreelayernetwork(X_train.shape[1], num_nodes, "zeros", d, lr)
    nn_model.train(X_train, Y_train, T)

    printerrors(nn_model)

