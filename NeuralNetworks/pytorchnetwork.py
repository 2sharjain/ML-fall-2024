import torch
from torch import nn
import pandas as pd
import numpy as np

device = "cpu"
widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=4, width=128, depth=4, activation_fn=nn.ReLU()):
        super(NeuralNetwork, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(nn.Linear(input_size, width),activation_fn)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(width, width),activation_fn) for _ in range(depth - 2)])
        
        # Output layer
        self.output_layer = nn.Linear(width, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def test(model, X, Y):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), Y.to(device)
        pred = model(X.float())
        pred = (pred > 0.5).float()
        error = torch.mean((pred != y).float())

    return error


def xavier(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        model.bias.data.fill_(0.01)


def he(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0.01)



activations = [(nn.ReLU(), he, "ReLU"), (nn.Tanh(), xavier, "Tanh")]

for ac_fn, func, ac_name in activations:
    for width in widths:
        for depth in depths:

            print(f"{depth}-depth, {width}-width network with activation function {ac_name}:\n")


            model = NeuralNetwork().to(device)
            model.apply(func)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_losses = np.array([])
            epochs = 20



            bank_train_df = pd.read_csv('bank-note/train.csv', header=None)
            bank_test_df = pd.read_csv('bank-note/test.csv', header=None)

            bank_train_df.columns = bank_test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']


            torch_trainX = torch.from_numpy(bank_train_df.iloc[:, :-1].values).float()
            torch_trainY = torch.from_numpy(bank_train_df.iloc[:, -1].values).float()
            torch_textX = torch.from_numpy(bank_test_df.iloc[:, :-1].values).float()
            torch_textY = torch.from_numpy(bank_test_df.iloc[:, -1].values).float()

            print()
            train_error = test(model, torch_trainX, torch_trainY)
            print(f"Training error: {train_error:.3f}\n\n")
            test_error = test(model, torch_textX, torch_textY)
            print(f"Testing error: {test_error:.3f}\n\n\n")