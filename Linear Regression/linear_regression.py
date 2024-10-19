import numpy as np
import pandas as pd
from gd_impl import *
import sys
sys.path.append("../EnsembleLearning")
from EnsembleLearning import plotter

train_data = "concrete/train.csv"
test_data = "concrete/test.csv"

features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'output']

train_df = pd.read_csv(train_data, names=features).astype(float)
test_df = pd.read_csv(test_data, names=features).astype(float)

X_train = train_df.drop('output', axis=1)
y_train = train_df['output']

X_test = test_df.drop('output', axis=1)
y_test = test_df['output']

batch_gradient_descent = Batch_Gradient_Descent()

stochastic_gradient_descent = Stochastic_Gradient_Descent()

bgd_training_cost = batch_gradient_descent.optimize(X_train, y_train)
batch_gradient_descent_error = batch_gradient_descent.bgd_loss_func(X_test, y_test)
print('Batch Gradient Descent weights: ' + str(batch_gradient_descent.weight_vec))

sgd_training_cost = stochastic_gradient_descent.optimize(X_train, y_train)
stochastic_gradient_descent_error = stochastic_gradient_descent.sgd_loss_func(X_test, y_test)
print('Stochastic Gradient Descent weights: ' + str(stochastic_gradient_descent.weight_vec))







plotter.plotthisformeletmego('Batch Gradient Descent', 'epochs', 'Cost function value', bgd_training_cost, None, None)
plotter.plotthisformeletmego('Stochastic Gradient Descent','epochs', 'Cost function value', sgd_training_cost, None, None, 2)

analytical_gradient_descent = Analytical_Solution(X_train, y_train)
analytical_gradient_descent_error = analytical_gradient_descent.analytical_loss_func(X_test, y_test)
print('Analytical Gradient Descent weight: ' + str(analytical_gradient_descent.weight_vec))
print('Analytical Gradient Descent Error: ' + str(analytical_gradient_descent_error))

print('Batch Gradient Descent Weight Error: ' + str(np.linalg.norm(batch_gradient_descent.weight_vec - analytical_gradient_descent.weight_vec)))
print('Stochastic Gradient Descent Weight Error: ' + str(np.linalg.norm(stochastic_gradient_descent.weight_vec - analytical_gradient_descent.weight_vec)))

