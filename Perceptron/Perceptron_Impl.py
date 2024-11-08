import random
import numpy as np

class StandardPerceptron():
    def train(self, X,Y,epochs, lr=0.1):
        weights = np.zeros(X.shape[1])
        data_length=len(X)
        for epoch in range(epochs):
            # np.random.seed(epoch)
            i = np.random.permutation(data_length)
            for j in range(data_length):
                if Y[i][j] * np.dot(weights,X[i][j]) <= 0:
                    weights += lr*X[i][j]*Y[i][j]
        return weights
    
    def predict(self, X, weights):
        return np.sign(np.dot(X, weights))
    
    def evaluate(self, X, Y, weights):
        return np.mean(self.predict(X,weights) != Y)
    
class VotedPerceptron():
    def train(self, X, Y, epochs, lr=0.1):
        weights= np.zeros(X.shape[1])
        weights_list = []
        votes = []
        vote = 1
        weights_list.append(weights)
        votes.append(vote)
        data_length = len(X)

        for epoch in range(epochs):
            # np.random.seed(epoch)
            i = np.random.permutation(data_length)
            for j in range(data_length):
                if Y[i][j] * np.dot(weights,X[i][j]) <= 0:
                    weights += lr*X[i][j]*Y[i][j]
                    weights_list.append(weights)
                    votes.append(vote)
                    vote=1
                else:
                    vote += 1

        return weights_list, votes

    def predict(self, X, weights, votes):
        return np.sign(np.dot(votes, np.sign(np.dot(X, weights.T)).T))
    
    def evaluate(self, X, Y, weights, votes):
        return np.mean(self.predict(X, weights, votes) != Y)
    
class AveragePerceptron():
    def train(self, X,Y,epochs, lr=0.1):
        weights = np.zeros(X.shape[1])
        a = weights
        data_length=len(X)
        for epoch in range(epochs):
            # np.random.seed(epoch)
            i = np.random.permutation(data_length)
            for j in range(data_length):
                if Y[i][j] * np.dot(weights,X[i][j]) <= 0:
                    weights += lr*X[i][j]*Y[i][j]
                    a += weights
        return a
    
    def predict(self, X, weights):
        return np.sign(np.dot(X, weights))
    
    def evaluate(self, X, Y, weights):
        return np.mean(self.predict(X,weights) != Y)
    

class AveragePerceptronNorm():
    def train(self, X,Y,epochs, lr=0.1):
        weights = np.zeros(X.shape[1])
        a = weights
        data_length=len(X)
        for epoch in range(epochs):
            # np.random.seed(epoch)
            i = np.random.permutation(data_length)
            for j in range(data_length):
                if Y[i][j] * np.dot(weights,X[i][j]) <= 0:
                    weights += lr*X[i][j]*Y[i][j]
                    a += weights
                    norm=np.linalg.norm(a)
                    a = a/norm

        return a
    
    def predict(self, X, weights):
        return np.sign(np.dot(X, weights))
    
    def evaluate(self, X, Y, weights):
        return np.mean(self.predict(X,weights) != Y)