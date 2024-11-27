import numpy as np
from scipy.optimize import minimize, Bounds

class PrimalSVM:
    def __init__(self, lr_type, a, bias=0, lr=0.001, C=1.0):
        self.C = C
        self.w = None
        self.a = a
        self.bias = bias
        if lr_type == "lr_a":
            self.lr_inc = self.learning_rate_increase_on_a
        elif lr_type == "lr_epoch":
            self.lr_inc = self.learning_rate_increase_on_epoch

        self.learning_rate = lr

    def fit(self, X, Y, epochs=100):
        # add bias
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X = np.hstack((X, np.zeros((X.shape[0], 1))))

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for epoch in range(epochs):
            lr_epoch = self.lr_inc(epoch)

            index = np.random.permutation(n_samples)
            X = X[index]
            Y = Y[index]

            for i in range(n_samples):
                xi = X[i]
                yi = Y[i]
                if yi * np.dot(xi, self.w) <= 1:
                    dw = np.append(self.w[:len(self.w) - 1], 0) - self.C * n_samples * yi * xi
                    self.w = self.w - lr_epoch * dw
                else:
                    self.w[:len(self.w) - 1] = (1 - lr_epoch) * self.w[:len(self.w) - 1]

    def learning_rate_increase_on_epoch(self, epoch):
        return self.learning_rate / (1 + epoch)

    def learning_rate_increase_on_a(self, epoch):
        return self.learning_rate / (1 + (self.learning_rate * epoch) / self.a)

    def predict(self, X):
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        else:
            X = np.hstack((X, np.zeros((X.shape[0], 1))))

        return np.sign(np.dot(X, self.w))

    def evaluate(self, x: np.ndarray, Y: np.ndarray):
        return np.mean(self.predict(x) != Y)
    


class DualSVM:
    def __init__(self, kernel_type, gamma = 0.0, C=1.0):
        self.C = C
        self.lambdas = None
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.gamma = gamma
        self.sup_vecs = None
        self.overlapping_sup_vecs = None
        if kernel_type == "linear":
            self.kernel = self.linear_kernel
        elif kernel_type == "gaussian":
            self.kernel = self.gaussian_kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        self.lambdas = np.zeros(n_samples)


        def objective_function(lambdas):
            out = -np.sum(lambdas) + 0.5 * np.dot(self.lambdas, np.dot(self.lambdas.T, (self.y@self.y.T) * self.kernel(self.X, self.X)))
            return out


        constraints = ({'type': 'eq', 'fun': self.constraints})
        bounds = Bounds(0, self.C)

        initial_guess = np.zeros(n_samples)


        solution = minimize(fun=objective_function, x0=initial_guess, bounds=bounds, method='SLSQP', constraints=constraints)
        self.lambdas = solution.x
        self.sup_vecs = np.where(self.lambdas > 1e-5)[0]

        self.overlapping_sup_vecs = np.where((self.lambdas > 1e-5) & (self.lambdas < self.C))[0]

        self.w = np.dot(self.lambdas * self.y, self.X)
        self.b = np.dot(self.lambdas, self.y)

    def constraints(self, lambdas):
        return np.dot(lambdas.T, self.y)

    def predict(self, X):
        prediction_res = []

        for i in range(len(X)):
            prediction = np.sign(sum(self.lambdas[self.sup_vecs] * self.y[self.sup_vecs] * self.kernel(self.X[self.sup_vecs], X[i])))
            if prediction > 0:
                prediction_res.append(1)
            else:
                prediction_res.append(-1)

        return np.array(prediction_res)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) != y)

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def gaussian_kernel(self, x1: np.ndarray, x2: np.ndarray):
        return np.exp(-np.linalg.norm(x1-x2)**2 / self.gamma)