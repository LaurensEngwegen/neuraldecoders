import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=2, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout_rate)

        self.linear = nn.Linear(n_hidden, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        output = self.linear(hidden_state[-1]) # Last layer, last state
        output = self.softmax(output)
        return output

class LSTM_Classifier():
    def __init__(self, X, y, labelsdict, 
                 n_hidden=256, 
                 n_layers=2, 
                 dropout_rate=0.5,
                 loss_fct=nn.CrossEntropyLoss,
                 optimizer=optim.Adam):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.X = torch.from_numpy(X.reshape(X.shape[0], X.shape[2], X.shape[1])).to(self.device)
        print(self.X.shape)
        self.y = y
        self.labels = []
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
            self.y = np.where(y==id, i, self.y)
        self.y = np.eye(len(np.unique(self.y)))[self.y]
        self.y = torch.from_numpy(self.y).to(self.device)
        self.id2label = labelsdict
        print(self.y.shape)
        self.n_classes = len(self.labels)
        self.n_features = X.shape[0] # nr electrodes
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.loss_fct = loss_fct
        self.optimizer = optimizer


    def train(self, X_train, y_train, n_epochs=15, verbose=0):
        self.model.train()
        # Initialize optimizer and loss function
        optimizer = self.optimizer(self.model.parameters())
        lossfunction = self.loss_fct()
        # Start training for n_epochs
        for epoch in range(n_epochs):
            if verbose:
                print(f'\nEpoch {epoch+1}/{n_epochs}...')
            epoch_loss = 0
            for X, y in zip(X_train, y_train):
                optimizer.zero_grad()
                print(X)
                X = X[None, :] # Needed when unbatched
                y = y[None, :]
                # y = argmax eerst
                print(X)
                output = self.model.forward(X)
                # print(output)
                output = output[None, :]
                print('\n\n\n')
                print(output)
                print(y)
                print(torch.argmax(output))
                print(torch.argmax(y))
                loss = lossfunction(output, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            epoch_loss = epoch_loss/X_train.shape[0]
            if verbose:
                print(f'Avg. loss = {epoch_loss}')

    def single_classification(self, test_index):
        self.model = LSTMModel(self.n_features, self.n_classes, self.n_hidden, self.n_layers, self.dropout_rate)
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        self.model.eval()
        test_X = self.X[test_index]
        test_X = test_X[None, :]
        y_pred = torch.argmax(self.forward(test_X))
        y_true = torch.argmax(self.y[test_index])
        print(f'Predicted: {y_pred}, true: {y_true}')

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
        for i in range(len(self.y)):
            print(f'Trial {i+1}/{len(self.y)}...')
            self.model = LSTMModel(self.n_features, self.n_classes, self.n_hidden, self.n_layers, self.dropout_rate)
            train_indices = [j for j in range(len(self.y)) if j != i]
            self.train(self.X[train_indices], self.y[train_indices])
            self.model.eval()
            test_X = self.X[i]
            test_X = test_X[None, :]
            y_pred = torch.argmax(self.forward(test_X))
            y_true = torch.argmax(self.y[i])
            y_preds.append(y_pred)
            y_trues.append(y_true)
            print(f"{i}. True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
        print(f'Accuracy = {correct/len(self.y)}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues.cpu(), y_preds.cpu(), display_labels=self.labels)
            plt.show()
        return correct/len(self.y)