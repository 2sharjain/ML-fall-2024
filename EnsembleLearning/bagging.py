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


class Bagged_Tree:
    def __init__(self, training_data, testing_data, features, labels, max_depth, num_trees, benchmark='entropy'):
        self.training_data = training_data
        self.testing_data = testing_data
        self.features = features
        self.labels = labels
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.benchmark = benchmark
        self.trees = []
        self.training_error, self.testing_error = self.build_tree()

    def build_tree(self):

        training_error = []
        testing_error = []
        subset_size = len(self.training_data)

        for _ in tqdm(range(self.num_trees)):

            bootstrap = self.training_data.sample(n=subset_size, replace=True)
            tree = DecisionTree(bootstrap, self.features, bootstrap['y'], self.max_depth, self.benchmark)
            self.trees.append(tree)

            training_error.append(self.evaluate(self.training_data, 'y'))
            testing_error.append(self.evaluate(self.testing_data, 'y'))

        return training_error, testing_error

    def predict(self, row):
        predictions = []

        for tree in self.trees:
            predictions.append(tree.predict(row))

        return max(set(predictions), key=predictions.count)

    def predictions(self, data):
        return data.apply(self.predict, axis=1)

    def evaluate(self, data: pd.DataFrame, label: str):
        predictions = self.predictions(data)
        actual = data[label]
        return np.mean(predictions != actual)
    



#script

training_data = "bank/train.csv"
testing_data = "bank/test.csv"

bank_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                     'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

bank_numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

train_data_frame = pd.read_csv(training_data, names=bank_columns)
test_data_frame = pd.read_csv(testing_data, names=bank_columns)

train_data_frame['y'] = train_data_frame['y'].apply(lambda x: '1' if x == 'yes' else '-1')
train_data_frame['y'] = train_data_frame['y'].astype(float)

test_data_frame['y'] = test_data_frame['y'].apply(lambda x: '1' if x == 'yes' else '-1')
test_data_frame['y'] = test_data_frame['y'].astype(float)


train_numerical_thresholds = train_data_frame[bank_numerical_columns].median()
test_numerical_thresholds = test_data_frame[bank_numerical_columns].median()


preprocessed_bank_train_df = preprocess_bank_data(train_data_frame, train_numerical_thresholds,bank_numerical_columns)
preprocessed_bank_test_df = preprocess_bank_data(test_data_frame, test_numerical_thresholds, bank_numerical_columns)

print("Bagged Tree Performance for Bank Dataset:")
Iteration = 500

bagged_tree = Bagged_Tree(preprocessed_bank_train_df, preprocessed_bank_test_df, list(preprocessed_bank_train_df.columns[:-1]),
                 preprocessed_bank_train_df['y'], 100, Iteration)

range_of_trees = range(1, 501)

plotthisformeletmego('Bagged Decision Tree Error', 'Number of trees', 'error', bagged_tree.training_error, bagged_tree.testing_error, 'Bagged Decision Tree Error.png')
