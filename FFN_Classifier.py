import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

class FFNModel(nn.Module):
    def __init__(self, n_features, n_classes, n_nodes=[128,64]):
        super(FFNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(n_features, n_nodes[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(n_nodes)):
            layers.append(nn.Linear(n_nodes[i-1], n_nodes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_nodes[i], n_classes))
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class FFN_Classifier():
    def __init__(self, X, y, labelsdict, 
                 n_nodes=[512,256],
                 batch_size=1,
                 loss_fct=nn.CrossEntropyLoss,
                 optimizer=optim.Adam):
        # Search for cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Reshape to [samples, features] and convert to float32 (as model weights are float32)
        X_second_dim = 1
        for i in range(1, len(X.shape)):
            X_second_dim *= X.shape[i]
        self.X = X.reshape(X.shape[0], X_second_dim).astype(np.float32)
        self.labels = []
        self.y = y.astype(np.long)
        for i, (_, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
        # Convert labels to labels from 0 to (#classes-1)
        for i in range(len(y)):
            index = y[i]
            label = labelsdict[index]
            self.y[i] = self.labels.index(label)
        # Data properties
        self.id2label = labelsdict
        self.n_classes = len(self.labels)
        self.n_features = self.X.shape[1]
        # Hyperparameters
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.loss_fct = loss_fct
        self.optimizer = optimizer

    def create_dataloader(self, X, y):
        # Need shape [examples, timesteps, features]
        # X = np.transpose(X, (0,2,1))
        X = torch.from_numpy(X).to(self.device)
        # Need to one-hot encode y
        # y = np.eye(self.n_classes)[y]
        y = torch.from_numpy(y).to(self.device)
        # Create dataloader
        dataloader = DataLoader(TensorDataset(X, y), shuffle=False, batch_size=self.batch_size)
        return dataloader, X, y

    def train(self, dataloader, n_epochs=10, verbose=0):
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
        self.model = FFNModel(self.n_features, self.n_classes, self.n_nodes).to(self.device)
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        train_dataloader, self.X_train, self.y_train = self.create_dataloader(self.X[train_indices], self.y[train_indices])
        print(self.model)
        self.train(train_dataloader, n_epochs=30)

        self.model.eval()
        _, X_test, y_test = self.create_dataloader(self.X[test_index:test_index+1], self.y[test_index:test_index+1])
        y_pred = torch.argmax(self.model.forward(X_test), dim=1).item()
        y_true = y_test.item()
        print(f"True - predicted ==> {y_true} - {y_pred}")

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
    
        for i in tqdm(range(len(self.y))):
            # print(f'\nTrial {i+1}/{len(self.y)}...')
            self.model = FFNModel(self.n_features, self.n_classes, self.n_nodes).to(self.device)
            train_indices = [j for j in range(len(self.y)) if j != i]
            train_dataloader, self.X_train, self.y_train = self.create_dataloader(self.X[train_indices], self.y[train_indices])
            
            self.train(train_dataloader, n_epochs=20)

            self.model.eval()
            _, X_test, y_test = self.create_dataloader(self.X[i:i+1], self.y[i:i+1])
            y_pred = torch.argmax(self.model.forward(X_test), dim=1).item()
            y_true = y_test.item()
            y_preds.append(y_pred)
            y_trues.append(y_true)
            # print(f"True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
        accuracy = correct/len(self.y)
        print(f'LOO Accuracy = {correct/len(self.y)}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy