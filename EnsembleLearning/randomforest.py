import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append("../DecisionTree")
from DecisionTree.preprocessor import *
from plotter import *


class RandomDecisionTree:
    def __init__(self, data, attributes, labels, max_depth, subset_size, criterion='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.criteria = criterion
        self.subset_size = subset_size
        self.tree = self.build_tree(data, attributes, labels)

    def build_tree(self, data: pd.DataFrame, attributes: list, labels: pd.Series, depth=0):
        # If all examples have same label, return the leaf node with the label
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]

        # If there are no more attributes to split on, return the most common label(target) value
        if len(attributes) == 0:
            return np.unique(labels).tolist()[0]

        # If the depth of the tree is equal to the maximum depth, return the most common label(target) value
        if depth == self.max_depth:
            return np.unique(labels).tolist()[0]

        # get samples uniformly with replacement
        samples = data.sample(n=self.subset_size, replace=True)

        best_attribute = self.choose_attribute(samples, samples['y'], attributes)
        tree = {best_attribute: {}}  # Create a root node

        for value in set(data[best_attribute]):
            # Split the data and target based on the value of the best attribute
            new_data = data[data[best_attribute] == value]
            new_label = labels[data[best_attribute] == value]
            # Remove the best attribute from the list of attributes
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            # Create a subtree for the current value of the attribute
            subtree = self.build_tree(new_data, new_attributes, new_label, depth + 1)
            # Add the subtree to the tree under the root node
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, data: pd.DataFrame, labels: pd.Series, attributes: list):
        # Calculate the information gain for each attribute
        gains = []

        for attribute in attributes:
            gains.append(self.information_gain(data, labels, attribute))

        # Return the attribute with the maximum information gain
        return attributes[gains.index(max(gains))]

    def information_gain(self, data: pd.DataFrame, labels: pd.Series, attribute: str):
        # Calculate the first term of Gain of the target values
        if self.criteria == 'entropy':
            first_term = self.entropy(labels)
        elif self.criteria == 'gini':
            first_term = self.gini_index(labels)
        else:
            first_term = self.majority_error(labels)

        # Calculate the values and the corresponding counts for the selected attribute
        values, counts = np.unique(data[attribute], return_counts=True)
        # Calculate the weighted entropy for the attribute
        second_term = 0

        for value, count_val in zip(values, counts):
            if self.criteria == 'entropy':
                second_term += (count_val / len(data)) * self.entropy(labels[data[attribute] == value])
            elif self.criteria == 'gini':
                second_term += (count_val / len(data)) * self.gini_index(labels[data[attribute] == value])
            else:
                second_term += (count_val / len(data)) * self.majority_error(labels[data[attribute] == value])

        # Calculate the information gain
        return first_term - second_term

    def entropy(self, label: pd.Series):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the entropy
        entropy = 0

        for count in counts:
            entropy += (-count / len(label)) * np.log2(count / len(label))

        return entropy

    def gini_index(self, label):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the gini index
        gini = 1

        for count in counts:
            gini -= (count / len(label)) ** 2

        return gini

    def majority_error(self, label):
        # Calculate the frequency of each unique value in the target
        _, counts = np.unique(label, return_counts=True)
        # Calculate the majority error
        majority_error = 1 - max(counts) / len(label)

        return majority_error

    def predict(self, row):
        """Predict the label of a row"""
        node = self.tree  # Start at the root node

        while isinstance(node, dict):  # While the node is not a leaf
            attribute = list(node.keys())[0]  # Get the attribute
            attribute_value = row[attribute]  # Get the value of the attribute

            if attribute_value not in node[attribute].keys():
                return None

            node = node[attribute][attribute_value]  # Move to the next node

        return node  # Return the leaf node

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)

    def training_error(self, label: str):
        """Calculate the training error of the decision tree"""
        return self.evaluate(self.data, label)

class RandomForest:
    def __init__(self, train_data, test_data, attributes, labels, max_depth, num_trees, subset_size,
                 criterion='entropy'):
        self.train_data = train_data
        self.test_data = test_data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = min(len(attributes), max_depth)
        self.subset_size = subset_size
        self.num_trees = num_trees
        self.criteria = criterion
        self.trees = []
        self.train_error, self.test_error = self.build_trees()

    def build_trees(self):
        train_error = []
        test_error = []
        sample_subset_size = len(self.train_data)

        for _ in tqdm(range(self.num_trees)):
            # get samples uniformly with replacement
            samples = self.train_data.sample(n=sample_subset_size, replace=True)
            # build a decision tree
            tree = RandomDecisionTree(samples, self.attributes, samples['y'], self.max_depth, self.subset_size,
                                      self.criteria)
            # add the tree to the forest
            self.trees.append(tree)
            # calculate the training error
            train_error.append(self.evaluate(self.train_data, 'y'))
            # calculate the test error
            test_error.append(self.evaluate(self.test_data, 'y'))

        return train_error, test_error

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def predict(self, row):
        """Predict the label of a row"""
        predictions = [tree.predict(row) for tree in self.trees]
        return max(set(predictions), key=predictions.count)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # average prediction error
        return np.mean(predictions != actual)

train_data = "bank/train.csv"
test_data = "bank/test.csv"

bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

# bank data numerical columns
bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_df = pd.read_csv(train_data, names=bank_column_names)
test_df = pd.read_csv(test_data, names=bank_column_names)

train_df['y'] = train_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
test_df['y'] = test_df['y'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

# median of numerical attributes of bank train data
train_numerical_thresholds = train_df[bank_numerical_columns].median()
test_numerical_thresholds = test_df[bank_numerical_columns].median()

#  consider unknown as category
preprocessed_bank_train_df = preprocess_bank_data(train_df, train_numerical_thresholds,
                                                            bank_numerical_columns)
preprocessed_bank_test_df = preprocess_bank_data(test_df, test_numerical_thresholds, bank_numerical_columns)

print("Random Forest Performance for Bank Dataset:")
T = 500
x = range(1, T + 1)

rf_2 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 2)
plotthisformeletmego('Random Forest Error Set Size 2', 'Number of trees', 'Error', rf_2.train_error, rf_2.test_error, 'Random Forest Error size 2.png',1)

rf_4 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 4)
plotthisformeletmego('Random Forest Error Set Size 4', 'Number of trees', 'Error', rf_4.train_error, rf_4.test_error, 'Random Forest Error size 4.png',2)

rf_6 = RandomForest(preprocessed_bank_train_df, preprocessed_bank_test_df,
                    list(preprocessed_bank_train_df.columns[:-1]),
                    preprocessed_bank_train_df['y'], 16, T, 6)
plotthisformeletmego('Random Forest Error Set Size 6', 'Number of trees', 'Error', rf_6.train_error, rf_6.test_error, 'Random Forest Error size 6.png',3)
