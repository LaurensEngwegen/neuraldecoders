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
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        output = self.linear(hidden_state[-1]) # Last layer, last state
        # output = self.softmax(output)
        return output

class LSTM_Classifier():
    def __init__(self, X, y, labelsdict, 
                 n_hidden=64, 
                 n_layers=2, 
                 dropout_rate=0.2,
                 batch_size=1,
                 loss_fct=nn.CrossEntropyLoss,
                 optimizer=optim.Adam):
        # Search for cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Need to convert to float32 (as model weights are float32)
        self.X = X.astype(np.float32)
        self.labels = []
        self.y = y.astype(np.long)
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
        # Convert labels to labels from 0 to (#classes-1)
        for i in range(len(y)):
            index = y[i]
            label = labelsdict[index]
            self.y[i] = self.labels.index(label)
        # Data properties
        self.id2label = labelsdict
        self.n_classes = len(self.labels)
        self.n_features = X.shape[1] # nr electrodes
        # Hyperparameters for LSTM
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.loss_fct = loss_fct
        self.optimizer = optimizer

    def create_dataloader(self, X, y):
        # Need shape [examples, timesteps, features]
        X = np.transpose(X, (0,2,1))
        X = torch.from_numpy(X).to(self.device)
        # Need to one-hot encode y
        # y = np.eye(self.n_classes)[y]
        y = torch.from_numpy(y).to(self.device)
        # Create dataloader
        dataloader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=self.batch_size)
        return dataloader, X, y

    def train(self, dataloader, n_epochs=10, verbose=1):
        # Initialize optimizer and loss function
        optimizer = self.optimizer(self.model.parameters())
        lossfunction = self.loss_fct()
        # Start training for n_epochs
        for epoch in range(n_epochs):
            self.model.train()
            if verbose:
                print(f'\nEpoch {epoch+1}/{n_epochs}...')
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model.forward(batch_X)
                loss = lossfunction(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            if verbose:
                print(f'Loss (averaged over batches): {epoch_loss/len(dataloader)}')
                # Training accuracy
                self.model.eval()
                out = self.model.forward(self.X_train)
                y_pred = torch.argmax(out, dim=1)
                y_true = self.y_train
                correct = torch.sum(y_pred == y_true)
                acc = correct/self.y_train.shape[0]
                print(f'Training accuracy: {acc}')
            

    # TODO
    def single_classification(self, test_index):
        self.model = LSTMModel(self.n_features, self.n_classes, self.n_hidden, self.n_layers, self.dropout_rate).to(self.device)
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        train_dataloader, self.X_train, self.y_train = self.create_dataloader(self.X[train_indices], self.y[train_indices])
        print(self.model)
        self.train(train_dataloader, n_epochs=30)

        self.model.eval()
        _, X_test, y_test = self.create_dataloader(self.X[test_index:test_index+1], self.y[test_index:test_index+1])
        # y_pred = torch.argmax(self.model.forward(X_test), dim=1).item()
        # y_true = torch.argmax(y_test, dim=1).item()
        y_pred = torch.argmax(self.model.forward(X_test), dim=1).item()
        y_true = y_test.item()
        print(f"True - predicted ==> {y_true} - {y_pred}")

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
        # for i in range(1):
        #     self.single_classification(i)
        
        for i in tqdm(range(len(self.y))):
            # print(f'\nTrial {i+1}/{len(self.y)}...')
            self.model = LSTMModel(self.n_features, self.n_classes, self.n_hidden, self.n_layers, self.dropout_rate).to(self.device)
            train_indices = [j for j in range(len(self.y)) if j != i]
            train_dataloader, self.X_train, self.y_train = self.create_dataloader(self.X[train_indices], self.y[train_indices])
            
            self.train(train_dataloader, n_epochs=15)

            self.model.eval()
            _, X_test, y_test = self.create_dataloader(self.X[i:i+1], self.y[i:i+1])
            y_pred = torch.argmax(self.model.forward(X_test), dim=1).item()
            y_true = y_test.item()
            y_preds.append(y_pred)
            y_trues.append(y_true)
            print(f"True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
        accuracy = correct/len(self.y)
        print(f'LOO Accuracy = {correct/len(self.y)}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues.cpu(), y_preds.cpu(), display_labels=self.labels)
            plt.show()
        return accuracy, y_trues, y_preds
        