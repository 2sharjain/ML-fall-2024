import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append("../DecisionTree")
from DecisionTree.preprocessor import *
from DecisionTree.id3_decisiontree import DecisionTree
from plotter import *

class Ada_Boost:
    def __init__(self, training_data, testing_data, attributes, labels, max_depth, num_trees, criterion='entropy'):
        self.training_data = training_data
        self.testing_data = testing_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.training_error_decision_tree, self.test_error_decision_tree, self.training_error, self.testing_error = self.build_trees(training_data, testing_data, attributes, labels, num_trees)

    def build_trees(self, training_data: pd.DataFrame, testing_data: pd.DataFrame, attributes: list, labels: pd.Series,
                    num_trees: int):
        weights = np.ones(len(training_data)) / len(training_data) #uniform
        training_error = []
        testing_error = []
        training_error_decision_tree = []
        test_error_decision_tree = []

        for _ in tqdm(range(num_trees)):
            # print(f" sum of weights = {np.sum(weights)}")

            # Create a decision tree
            # get samples uniformly with replacement
            samples = self.training_data.sample(n=self.training_data.shape[0], replace=True, weights=weights)
            # build a decision tree
            tree = DecisionTree(samples, self.attributes, samples['y'], self.max_depth, self.criteria)
            # Calculate the predictions of the decision tree
            predictions = tree.predictions(training_data)
            print(f"Predictions at iteration {_}: {predictions[:5]}")
            # Calculate the error of the decision tree
            mismatch_mask = np.where(predictions != labels, 1, 0)
            error = np.sum(weights * mismatch_mask)
            print(f"error at iteration{_} is {error}")
            # Calculate the weight of the decision tree
            weight = 0.5 * np.log((1 - error) / error)
            print(f"weight at iteration{_} is {weight}")

            # Update the weights of the examples
            weights = weights * np.exp(-weight * labels * predictions)
            # print(f"weights={np.median(weights)}")

            # Normalize the weights
            weights = weights / np.sum(weights)
            # Add the tree to the forest
            self.trees.append((tree, weight))
            # Calculate the training and testing error
            training_error_decision_tree.append(tree.evaluate(training_data, 'y'))
            # print(tree.evaluate(training_data, 'y'))
            test_error_decision_tree.append(tree.evaluate(testing_data, 'y'))
            training_error.append(self.evaluate(training_data, 'y'))
            testing_error.append(self.evaluate(testing_data, 'y'))
        
        
        # print(f"weights={weights}")

        print(training_error_decision_tree)
        print(training_error)
        return training_error_decision_tree, test_error_decision_tree, training_error, testing_error

    def predict(self, row):
        """Predict the label of a row"""
        predictions = []

        for tree, weight in self.trees:
            predictions.append(weight * tree.predict(row))

        return np.sign(sum(predictions))

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)




####   start of script   ####


training_data = "bank/train.csv"
testing_data = "bank/test.csv"

columns_bank = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

numerical_columns_bank = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

training_df = pd.read_csv(training_data, names=columns_bank)
testing_df = pd.read_csv(testing_data, names=columns_bank)

training_df['y'] = training_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
testing_df['y'] = testing_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

# median
train_numerical_thresholds_bank = training_df[numerical_columns_bank].median()
test_numerical_thresholds_bank = testing_df[numerical_columns_bank].median()

preprocessed_bank_train_df = preprocess_bank_data(training_df, train_numerical_thresholds_bank,
                                                            numerical_columns_bank)
preprocessed_bank_test_df = preprocess_bank_data(testing_df, test_numerical_thresholds_bank, numerical_columns_bank)

print("Bank eval with unknown as a value:")
iterations = 500

ada = Ada_Boost(preprocessed_bank_train_df, preprocessed_bank_test_df, list(preprocessed_bank_train_df.columns[:-1]),
               preprocessed_bank_train_df['y'], 1, iterations)




plotthisformeletmego('Tree Prediction Error', 'Iteration', 'Error Rate', ada.training_error_decision_tree, ada.test_error_decision_tree, 'tree prediction error.png',1)
plotthisformeletmego('Adaboost Error', 'Iteration', 'Error Rate', ada.training_error, ada.testing_error, 'Adaboost prediction error.png',2)
