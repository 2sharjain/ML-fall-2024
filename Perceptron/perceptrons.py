import pandas as pd
import numpy as np

from Perceptron_Impl import StandardPerceptron, VotedPerceptron, AveragePerceptron, AveragePerceptronNorm

bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

cols = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
bank_train_df.columns = cols
bank_test_df.columns = cols


X_train = bank_train_df.iloc[:, :-1].values
Y_train = bank_train_df.iloc[:, -1].values
Y_train[Y_train == 0] = -1

X_test = bank_test_df.iloc[:, :-1].values
Y_test = bank_test_df.iloc[:, -1].values
Y_test[Y_test == 0] = -1

epochs=10
learning_rate = 0.1

standard_perceptron = StandardPerceptron()
standard_weights = standard_perceptron.train(X_train, Y_train, epochs, learning_rate)
standard_preceptron_error = standard_perceptron.evaluate(X_test, Y_test, standard_weights)

print("-------------STANDARD PERCEPTRON--------------")
print(f"Learned weight vector:  {standard_weights}")
print(f"Average prediction error is {standard_preceptron_error} on test dataset")
print("-------------------------------------------")
print("\n\n")


voted_perceptron = VotedPerceptron()
_weights_list, _votes = voted_perceptron.train(X_train, Y_train, epochs, learning_rate)
voted_weights_list = np.array(_weights_list)
votes = np.array(_votes)
voted_perceptron_error = voted_perceptron.evaluate(X_test, Y_test, voted_weights_list[1:], votes[1:])

print("-------------VOTED PERCEPTRON--------------")
print(f"Learned weight vector:  {voted_weights_list[-1]}")
print(f"Average prediction error is {voted_perceptron_error} on test dataset")
print("-------------------------------------------")
print("\n\n")
with open("voted_weights_and_votes.txt", "w") as f:
    for i in range(1, len(voted_weights_list)):
        f.write(f"Weight vector {i} : {voted_weights_list[i]} \t")
        f.write(f"Votes for weight vector {i} : {votes[i]} \n")
f.close()

average_perceptron = AveragePerceptron()
average_weights = average_perceptron.train(X_train, Y_train, epochs, learning_rate)
average_preceptron_error = average_perceptron.evaluate(X_test, Y_test, average_weights)
print("-------------AVERAGE PERCEPTRON--------------")
print(f"Learned weight vector:  {average_weights}")
print(f"Average prediction error is {average_preceptron_error} on test dataset")
print("-------------------------------------------")
print("\n\n")


average_perceptron_norm = AveragePerceptronNorm()
average_weights_norm = average_perceptron_norm.train(X_train, Y_train, epochs, learning_rate)
average_preceptron_error_norm = average_perceptron_norm.evaluate(X_test, Y_test, average_weights_norm)
print("-------------AVERAGE PERCEPTRON(normalized)--------------")
print(f"Learned weight vector:  {average_weights_norm}")
print(f"Average prediction error is {average_preceptron_error_norm} on test dataset")
print("-------------------------------------------")
print("\n\n")