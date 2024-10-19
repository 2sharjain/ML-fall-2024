import pandas as pd
import numpy as np

from utils import *

DEPTH = 17

class DecisionTree:
    def __init__(self, data, attributes, labels, max_depth, benchmark='entropy'):
        self.data = data
        self.attributes = attributes
        self.labels = labels
        self.max_depth = max_depth
        self.benchmark = benchmark
        self.tree = self.build_tree(data, attributes, labels)

    def build_tree(self, data: pd.DataFrame, attributes: list, labels: pd.Series, depth=0):

        if len(np.unique(labels)) == 1:
            return labels.iloc[0]

        if len(attributes) == 0:
            return np.unique(labels).tolist()[0]

        if depth == self.max_depth:
            return np.unique(labels).tolist()[0]

        best_attribute = self.choose_attribute(data, labels, attributes)
        tree = {best_attribute: {}}  # Create a root node

        for value in set(data[best_attribute]):
            new_data = data[data[best_attribute] == value]
            new_label = labels[data[best_attribute] == value]
            new_attributes = list(attributes[:])
            new_attributes.remove(best_attribute)
            subtree = self.build_tree(new_data, new_attributes, new_label, depth + 1)
            tree[best_attribute][value] = subtree

        return tree

    def choose_attribute(self, data: pd.DataFrame, labels: pd.Series, attributes: list):
        gains = []

        for attribute in attributes:
            gains.append(self.information_gain(data, labels, attribute))

        return attributes[gains.index(max(gains))]

    def information_gain(self, data: pd.DataFrame, labels: pd.Series, attribute: str):
        first_term = 0
        if self.benchmark == 'entropy':
            first_term = self.entropy(labels)
        elif self.benchmark == 'gini':
            first_term = self.gini_index(labels)
        elif self.benchmark == 'majority_error':
            first_term = self.majority_error(labels)

        values, counts = np.unique(data[attribute], return_counts=True)
        weighted_entropy = 0

        for value, count in zip(values, counts):
            if self.benchmark == 'entropy':
                weighted_entropy += (count / len(data)) * self.entropy(labels[data[attribute] == value])
            elif self.benchmark == 'gini':
                weighted_entropy += (count / len(data)) * self.gini_index(labels[data[attribute] == value])
            else:
                weighted_entropy += (count / len(data)) * self.majority_error(labels[data[attribute] == value])

        return first_term - weighted_entropy

    def entropy(self, label: pd.Series):
        _, counts = np.unique(label, return_counts=True)
        entropy = 0

        for count in counts:
            entropy = entropy+((-count / len(label)) * np.log2(count / len(label)))

        return entropy

    def gini_index(self, label):
        _, counts = np.unique(label, return_counts=True)
        gini = 1

        for count in counts:
            gini -= (count / len(label)) ** 2

        return gini

    def majority_error(self, label):
        _, counts = np.unique(label, return_counts=True)
        majority_error = 1 - max(counts) / len(label)

        return majority_error

    def predict(self, row):
        """Predict the label of a row"""
        node = self.tree  
        while isinstance(node, dict):  
            attribute = list(node.keys())[0]  
            attribute_value = row[attribute]  
            if attribute_value not in node[attribute].keys():
                return None

            node = node[attribute][attribute_value]  

        return node 

    def predictions(self, data):
        """Predict the labels of a dataset"""
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        """Evaluate the accuracy of the decision tree"""
        predictions = self.predictions(data)
        actual = data[label]
        # print(f" disparity = {np.size(predictions == actual)}")
        return np.mean(predictions != actual)

    def training_error(self, label: str):
        """Calculate the training error of the decision tree"""
        return self.evaluate(self.data, label)

