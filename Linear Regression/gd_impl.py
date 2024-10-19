import numpy as np
import pandas as pd
import random

class Stochastic_Gradient_Descent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weight_vec = None

    def optimize(self, x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        num_data, num_features = x_data.shape
        self.weight_vec = np.zeros(num_features)

        sgd_cost = []
        norm_weight_difference = 1

        while norm_weight_difference > 0.000001:
            predict_y = np.dot(x_data, self.weight_vec)
            sgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))

            rand_index = random.randint(0, num_data - 1)
            y_rand_predicted = np.dot(x_data[rand_index], self.weight_vec)
            delta = np.dot(x_data[rand_index].T, (y_rand_predicted - y_data[rand_index]))

            norm_weight_difference = np.linalg.norm(self.weight_vec - (self.weight_vec - self.learning_rate * delta))
            self.weight_vec -= self.learning_rate * delta
            sgd_cost.append(sgd_loss)

        return sgd_cost

    def sgd_loss_func(self, x_data, y_data):
        predict_y = np.dot(x_data, self.weight_vec)
        sgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))
        return sgd_loss


class Analytical_Solution:

    def __init__(self,  x_data, y_data):
        self.weight_vec = np.dot(np.linalg.inv(np.dot(x_data.T, x_data)), np.dot(x_data.T, y_data))

    def analytical_loss_func(self, x_data, y_data):
        predict_y = np.dot(x_data, self.weight_vec)
        analytical_loss = 0.5 * np.sum(np.square(y_data - predict_y))
        return analytical_loss

class Batch_Gradient_Descent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weight_vec = None

    def optimize(self, x_data, y_data):

        num_data, num_features = x_data.shape
        self.weight_vec = np.zeros(num_features)

        bgd_cost = []
        norm_weight_difference = 1

        while norm_weight_difference > 0.000001:
            predict_y = np.dot(x_data, self.weight_vec)
            bgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))
            delta = np.dot(x_data.T, (predict_y - y_data))

            norm_weight_difference = np.linalg.norm(self.weight_vec - (self.weight_vec - self.learning_rate * delta))
            self.weight_vec -= self.learning_rate * delta
            bgd_cost.append(bgd_loss)

        return bgd_cost

    def bgd_loss_func(self, x_data, y_data):
        predict_y = np.dot(x_data, self.weight_vec)
        bgd_loss = 0.5 * np.sum(np.square(y_data - predict_y))
        return bgd_loss