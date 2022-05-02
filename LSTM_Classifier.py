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
        self.softmax = F.softmax()

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        output = self.linear(hidden_state[-1]) # Last layer, last state
        output = self.softmax(output)
        return output

class LSTM_Classifier():
    def __init__(self, x, y, labelsdict, 
                 n_hidden=256, 
                 n_layers=2, 
                 dropout_rate=0.5,
                 loss_fct=nn.CrossEntropyLoss,
                 optimizer=optim.Adam):
        
        n_features = x.shape[0] # nr electrodes
        n_classes = len(np.unique(y)) # nr classes

        self.model = LSTMModel(n_features, n_classes, n_hidden, n_layers, dropout_rate)
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
                X = X[None, :] # Needed when unbatched
                y = y[None, :]
                # y = argmax eerst
                output = self.forward(X)
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
