import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

class constrainedConv2d(nn.Conv2d):
    # CALL THIS FUNCTION AFTER EACH OPTIMIZAITON STEP
    def max_norm(self, max_norm_val):
        norm = self.weight.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, max_norm_val)
        # print(f'\n\n\nmax_norm makes it:\n{self.weight * (desired / (1e-7 + norm))}\n')
        self.weight = nn.Parameter(self.weight * (desired / (1e-14 + norm)))

class constrainedLinear(nn.Linear):
    # CALL THIS FUNCTION AFTER EACH OPTIMIZAITON STEP
    def max_norm(self, max_norm_val):
        norm = self.weight.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, max_norm_val)
        # print(f'\n\n\nmax_norm makes it:\n{self.weight * (desired / (1e-7 + norm))}\n')
        self.weight = nn.Parameter(self.weight * (desired / (1e-14 + norm)))


class EEGNet_torch_Classifier(nn.Module):
    def __init__(self, X, y, labelsdict, 
                 n_channels,
                 n_samples,
                 dropoutRate=0.5,
                 kernLength=256,
                 F1=8,
                 D=2,
                 F2=16,
                 norm1=1,
                 norm2=0.25,
                 optimizer=optim.Adam, 
                 loss=nn.CrossEntropyLoss):
        super(EEGNet_torch_Classifier, self).__init__()
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device: {self.device}')
        
        self.X = torch.from_numpy(X).to(self.device)

        # self.y = np.zeros((len(y), 5)) # 5 classes
        # for i in range(len(y)):
        #     label = y[i]
        #     self.y[i,label-1] = 1
        # self.y = torch.from_numpy(self.y).to(self.device)
        self.y = y

        self.labels = []
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
            self.y = np.where(y==id, i, self.y)
        # print(self.y)
        self.y = np.eye(len(np.unique(self.y)))[self.y]
        self.y = torch.from_numpy(self.y).to(self.device)
        # print(self.y)
        self.id2label = labelsdict
        self.n_classes = len(self.labels)

        self.kernLength = kernLength
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropoutRate
        self.norm1 = norm1
        self.norm2 = norm2

        self.optimizer = optimizer
        self.loss = loss

    def initialize_model(self):
        
        self.layers = []

        # Temporal
        self.layers.append(nn.Conv2d(1, self.F1, (1, self.kernLength), padding='same', bias=False))
        # self.layers.append(nn.BatchNorm2d(F1, False))
        
        # Depthwise
        self.layers.append(constrainedConv2d(in_channels=self.F1, 
                                     out_channels=self.F1*self.D, 
                                     kernel_size=(self.n_channels, 1), 
                                     padding='valid', 
                                     groups=self.F1, 
                                     bias=False))
        # self.layers.append(nn.BatchNorm2d(..., False))
        self.layers.append(nn.ELU())
        self.layers.append(nn.AvgPool2d((1,4)))
        self.layers.append(nn.Dropout(self.dropout_rate))

        # Separable (= depthwise+pointwise)
        self.layers.append(nn.Conv2d(self.F1*self.D, 
                                     self.F1*self.D, 
                                     (1,16), 
                                     padding='same', 
                                     groups=self.F1*self.D, 
                                     bias=False))
        self.layers.append(nn.Conv2d(self.F1*self.D, 
                                     self.F2, 
                                     (1,1), 
                                     padding='same', 
                                     bias=False))
        # self.layers.append(nn.BatchNorm2d(..., False))
        self.layers.append(nn.ELU())
        self.layers.append(nn.AvgPool2d((1,8)))
        self.layers.append(nn.Dropout(self.dropout_rate))

        # Flatten, linear, softmax
        self.layers.append(nn.Flatten(0))
        self.layers.append(constrainedLinear(int(self.F2*self.n_samples/32), self.n_classes))
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
                X = X[None, :] # Needed when unbatched
                y = y[None, :]
                # y = argmax eerst
                output = self.forward(X)
                # print(output)
                output = output[None, :]
                loss = lossfunction(output, y)
                loss.backward()
                optimizer.step()
                self.layers[1].max_norm(self.norm1)
                self.layers[11].max_norm(self.norm2)

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
        return correct/len(self.y)