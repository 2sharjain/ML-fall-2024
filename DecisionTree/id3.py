import pandas as pd
import numpy as np
from id3_decisiontree import DecisionTree
from utils import *

DEPTH = 17

####### Decision Tree

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