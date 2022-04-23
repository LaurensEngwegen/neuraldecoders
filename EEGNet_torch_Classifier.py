import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm



class EEGNet_torch_Classifier(nn.Module):
    def __init__(self, X, y, labels, optimizer=optim.Adam, loss=nn.CrossEntropyLoss):
        super(EEGNet_torch_Classifier, self).__init__()
        '''
        self.T = samples_per_trial
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, sampling_rate/2), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)
        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device: {self.device}')
        
        self.X = torch.from_numpy(X).to(self.device)
        # TODO: make generalizable for more/less classes
        self.y = np.zeros((len(y), 5)) # 5 classes
        for i in range(len(y)):
            label = y[i]
            self.y[i,label-1] = 1
        self.y = torch.from_numpy(self.y).to(self.device)

        self.labels = labels
        self.id2label = dict()
        for i in range(len(self.labels)):
            self.id2label[i+1] = self.labels[i]

        self.optimizer = optimizer
        self.loss = loss


    def initialize_model(self,
                         n_classes = 5,
                         channels = 60,
                         sampling_rate = 512,
                         F1 = 8,
                         D = 2,
                         F2 = 16,
                         dropout_rate = 0.5):
        
        self.layers = []

        self.layers.append(nn.Conv2d(1, F1, (1, int(sampling_rate/2)), padding='same', bias=False))
        # self.layers.append(nn.BatchNorm2d(F1, False))
        
        # Depthwise
        # TODO: add depthwise_constraint = maxnorm(1.0)
        self.layers.append(nn.Conv2d(F1, F1*D, (channels, 1), padding='valid', groups=F1, bias=False))
        # self.layers.append(nn.BatchNorm2d(..., False))

        self.layers.append(nn.ELU())
        self.layers.append(nn.AvgPool2d((1,4)))
        self.layers.append(nn.Dropout(dropout_rate))

        # Separable (= depthwise+pointwise)
        self.layers.append(nn.Conv2d(F1*D, F1*D, (1,16), padding='same', groups=F1*D, bias=False))
        self.layers.append(nn.Conv2d(F1*D, F2, (1,1), padding='same', bias=False))
        # self.layers.append(nn.BatchNorm2d(..., False))

        self.layers.append(nn.ELU())
        self.layers.append(nn.AvgPool2d((1,8)))
        self.layers.append(nn.Dropout(dropout_rate))

        # Flatten, linear, softmax
        self.layers.append(nn.Flatten(0))
        # TODO: add kernel constraint
        self.layers.append(nn.Linear(256, n_classes))
        self.layers.append(nn.Softmax())

        self.model = nn.ModuleList(self.layers).double()

        self.model.to(self.device)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            # print(f'Layer: {layer}')
            # print(x.shape)
        return x

    # TODO: introduce possibility to train in batches
    def train(self, X_train, y_train, n_epochs=15, verbose=0):
        self.model.train()
        # Initialize optimizer and loss function
        optimizer = self.optimizer(self.model.parameters())
        lossfunction = self.loss()
        # Start training for n_epochs
        for epoch in range(n_epochs):
            if verbose:
                print(f'\nEpoch {epoch+1}/{n_epochs}...')
            epoch_loss = 0
            for X, y in zip(X_train, y_train):
                optimizer.zero_grad()
                X = X[None, :] # Needed when batchsize=1

                
                y = y[None, :]
                # y = argmax eerst

                output = self.forward(X)
                output = output[None, :]
                loss = lossfunction(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss/X_train.shape[0]
            if verbose:
                print(f'Avg. loss = {epoch_loss}')


    def predict(self, X_test):
        pass

    def single_classification(self, test_index):
        self.initialize_model()
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
            self.model = self.initialize_model()
            self.initialize_model()
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
