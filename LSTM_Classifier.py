import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
        self.softmax = nn.Softmax(dim=1)

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
                 batch_size=10,
                 loss_fct=nn.CrossEntropyLoss,
                 optimizer=optim.Adam):

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
        # Need to convert to float32 (as model weights are float32)
        X = X.astype(np.float32)
        self.labels = []
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
            y = np.where(y==id, i, y) # Map labels to 0 - (n_classes-1)
        self.id2label = labelsdict
        print(f'X shape: {X.shape}')
        print(f'y shape: {y.shape}')
        self.dataloader, self.X, self.y = self.create_dataloader(X, y, batch_size)
        print(f'X shape: {self.X.shape}')
        print(f'y shape: {self.y.shape}')

        self.n_classes = len(self.labels)
        self.n_features = X.shape[1] # nr electrodes
        print(f'n_feautures = {self.n_features}')
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.loss_fct = loss_fct
        self.optimizer = optimizer

    def create_dataloader(self, X, y, batch_size=1):
        # Need shape [examples, timesteps, features]
        X = np.transpose(X, (0,2,1))
        X = torch.from_numpy(X).to(self.device)
        # Need to one-hot encode y
        y = np.eye(len(np.unique(y)))[y]
        y = torch.from_numpy(y).to(self.device)
        # Create dataloader
        dataloader = DataLoader(TensorDataset(X, y), shuffle=False, batch_size=batch_size)
        return dataloader, X, y

    def train(self, n_epochs=10, verbose=1):
        self.model.train()
        # Initialize optimizer and loss function
        optimizer = self.optimizer(self.model.parameters())
        lossfunction = self.loss_fct()
        # Start training for n_epochs
        for epoch in range(n_epochs):
            if verbose:
                print(f'\nEpoch {epoch+1}/{n_epochs}...')
            epoch_loss = 0
            for batch_X, batch_y in tqdm(self.dataloader):
                optimizer.zero_grad()
                output = self.model.forward(batch_X)
                loss = lossfunction(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            print(f'Loss: {epoch_loss}')
            # epoch_loss = epoch_loss/X_train.shape[0]
            # if verbose:
            #     print(f'Avg. loss = {epoch_loss}')

    # TODO
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
        
        # Probably need to (re)create dataloader here,
        # after train and test set are defined (separate dataloader for train and test(?))
        
        y_preds, y_trues = [], []
        correct = 0
        self.model = LSTMModel(self.n_features, self.n_classes, self.n_hidden, self.n_layers, self.dropout_rate)
        self.train()
        self.model.eval()
        y_pred = self.model.forward(self.X)

        '''
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
        '''
        return 1