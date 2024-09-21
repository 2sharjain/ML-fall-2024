import pandas as pd
import numpy as np

from utils import *

DEPTH = 17

####### Decision Tree
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
        return np.mean(predictions != actual)

    def training_error(self, label: str):
        """Calculate the training error of the decision tree"""
        return self.evaluate(self.data, label)


#############










car_train_data = pd.read_csv('car/train.csv')
car_test_data = pd.read_csv('car/test.csv')

car_test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
car_train_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

df = pd.DataFrame(columns=["Depth", "Entropy_{train}", "Entropy_{test}", "Gini_{train}", "Gini_{test}", "Major_{train}", "Major_{test}"])

for depth in range(1, 6):
    for benchmark in ['entropy', 'gini', 'majority']:
        car_decision_tree = DecisionTree(car_train_data, list(car_train_data.columns[:-1]), car_train_data['label'],
                                         max_depth=depth, benchmark=benchmark)

        if benchmark == 'entropy':
            entropy_train = car_decision_tree.training_error('label')
            entropy_test = car_decision_tree.evaluate(car_test_data, 'label')

        elif benchmark == 'gini':
            gini_train = car_decision_tree.training_error('label')
            gini_test = car_decision_tree.evaluate(car_test_data, 'label')

        else:
            major_train = car_decision_tree.training_error('label')
            major_test = car_decision_tree.evaluate(car_test_data, 'label')

        # print(f"Car Dataset Errors:")
        # print(f"Depth:{depth} Benchmark:{benchmark} =>")
        # print(f"Average prediction training error: {car_decision_tree.training_error('label')}")
        # print(f"Average prediction testing error: {car_decision_tree.evaluate(car_test_data, 'label')}\n")

    df.loc[len(df)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

print(df.to_latex())

bank_train_data = pd.read_csv('bank/train.csv')
bank_test_data = pd.read_csv('bank/test.csv')
bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_train_data.columns = bank_column_names
bank_test_data.columns = bank_column_names

bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
train_numerical_thresholds = bank_train_data[bank_numerical_columns].median()
test_numerical_thresholds = bank_test_data[bank_numerical_columns].median()

bank_train_df = preprocessing_bank_data(bank_train_data, train_numerical_thresholds, bank_numerical_columns)
bank_test_df = preprocessing_bank_data(bank_test_data, test_numerical_thresholds, bank_numerical_columns)

df1 = pd.DataFrame(columns=["Depth", "Entropy_{train}", "Entropy_{test}", "Gini_{train}", "Gini_{test}", "Major_{train}", "Major_{test}"])

print("Bank Dataset Evaluation (with unknown considered as value):")
for depth in range(1, DEPTH):
    for benchmark in ['entropy', 'gini', 'majority']:
        bank_decision_tree = DecisionTree(bank_train_df, list(bank_train_df.columns[:-1]),
                                          bank_train_df['y'], max_depth=depth, benchmark=benchmark)

        if benchmark == 'entropy':
            entropy_train = bank_decision_tree.training_error('y')
            entropy_test = bank_decision_tree.evaluate(bank_test_df, 'y')

        elif benchmark == 'gini':
            gini_train = bank_decision_tree.training_error('y')
            gini_test = bank_decision_tree.evaluate(bank_test_df, 'y')

        else:
            major_train = bank_decision_tree.training_error('y')
            major_test = bank_decision_tree.evaluate(bank_test_df, 'y')

        # print(f"Bank Dataset Errors:")
        # print(f"Depth:{depth} Benchmark:{benchmark} =>")
        # print(f"Average prediction training error: {bank_decision_tree.training_error('y')}")
        # print(f"Average prediction testing error: {bank_decision_tree.evaluate(bank_test_df, 'y')}\n")


    df1.loc[len(df1)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

print(df1.to_latex())

categorical_columns_with_unknown_values = ['job', 'education', 'contact', 'poutcome']

bank_train_df = replace_unknown_data(bank_train_data, categorical_columns_with_unknown_values)
bank_test_df = replace_unknown_data(bank_test_data, categorical_columns_with_unknown_values)


df2 = pd.DataFrame(columns=["Depth", "Entropy_{train}", "Entropy_{test}", "Gini_{train}", "Gini_{test}", "Major_{train}", "Major_{test}"])

print("Bank Dataset Evaluation (with unknown replaced by most frequent value):")
for depth in range(1, DEPTH):
    for benchmark in ['entropy', 'gini', 'majority']:
        bank_decision_tree_for_replaced_unknown_values = DecisionTree(bank_train_df,
                                                                      list(bank_train_df.columns[:-1]),
                                                                      bank_train_df['y'], max_depth=depth,
                                                                       benchmark=benchmark)

        if benchmark == 'entropy':
            entropy_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            entropy_test = bank_decision_tree_for_replaced_unknown_values.evaluate(bank_test_df, 'y')

        elif benchmark == 'gini':
            gini_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            gini_test = bank_decision_tree_for_replaced_unknown_values.evaluate(bank_test_df, 'y')

        else:
            major_train = bank_decision_tree_for_replaced_unknown_values.training_error('y')
            major_test = bank_decision_tree_for_replaced_unknown_values.evaluate(bank_test_df, 'y')

        # print(f"Bank Dataset (replaced unknown value) Errors:")
        # print(f"Depth:{depth} Benchmark:{benchmark} =>")
        # print(f"Average prediction training error: {bank_decision_tree_for_replaced_unknown_values.training_error('y')}")
        # print(f"Average prediction testing error: {bank_decision_tree_for_replaced_unknown_values.evaluate(bank_test_df, 'y')}\n")

    df2.loc[len(df2)] = [depth, entropy_train, entropy_test, gini_train, gini_test, major_train, major_test]

print(df2.to_latex())