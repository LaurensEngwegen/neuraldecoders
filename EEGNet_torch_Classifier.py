import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

class EEGNet_Model(nn.Module):
    def __init__(self, 
                 n_channels,
                 n_samples,
                 n_classes,
                 dropout_rate=0.5,
                 kernLength=256,
                 F1=8,
                 D=2,
                 F2=16,):
        super(EEGNet_Model, self).__init__()

        # Temporal
        self.temporal = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        # self.layers.append(nn.BatchNorm2d(F1, False))

        # Depthwise
        self.depthwise = nn.Sequential(
            constrainedConv2d(in_channels=F1, 
                              out_channels=F1*D, 
                              kernel_size=(n_channels, 1), 
                              padding='valid', 
                              groups=F1, 
                              bias=False),
            # self.layers.append(nn.BatchNorm2d(..., False))
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(dropout_rate))

        # Separable (= depthwise+pointwise)
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D,
                      F1*D, 
                      (1,16), 
                      padding='same', 
                      groups=F1*D, 
                      bias=False),
            nn.Conv2d(F1*D,
                      F2, 
                      (1,1), 
                      padding='same',
                      bias=False),
            # self.layers.append(nn.BatchNorm2d(..., False))
            nn.ELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(dropout_rate))

        # Flatten, linear, softmax
        self.linear = nn.Sequential(
            nn.Flatten(),
            constrainedLinear(int(F2*n_samples/32), n_classes))
        # self.layers.append(nn.Softmax(dim=1))
        # model = nn.ModuleList(self.layers).float()
        # model.to(self.device)

    def forward(self, x):
        out = self.temporal(x)
        out = self.depthwise(out)
        out = self.separable(out)
        output = self.linear(out)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(f'Layer {layer}: {x.shape}')
        return output

class EEGNet_torch_Classifier():
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
                 batch_size=1,
                 optimizer=optim.Adam, 
                 loss=nn.CrossEntropyLoss):
        # Search for cuda device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Need to convert to float32 (as model weights are float32)
        self.X = X[:, np.newaxis, :, :].astype(np.float32)
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
        # Hyperparameters of EEGNet
        self.kernLength = kernLength
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropoutRate
        self.norm1 = norm1
        self.norm2 = norm2
        self.batch_size = batch_size
        # Optimizer and loss function
        self.optimizer = optimizer
        self.loss = loss

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

    def initialize_model(self):    
        pass



    # TODO: introduce possibility to train in batches
    def train(self, dataloader, n_epochs=15, verbose=0):
        # Initialize optimizer and loss function
        optimizer = self.optimizer(self.model.parameters())
        lossfunction = self.loss()
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
                self.model.depthwise[0].max_norm(self.norm1)
                self.model.linear[1].max_norm(self.norm2)
                epoch_loss += loss
            if verbose:
                print(f'Loss (averaged over batches): {epoch_loss/len(dataloader)}')
                # Training accuracy
                self.model.eval()
                out = self.model.forward(self.X_train)
                y_pred = torch.argmax(out, dim=1)
                # y_true = torch.argmax(self.y_train, dim=1)
                y_true = self.y_train
                correct = torch.sum(y_pred == y_true)
                acc = correct/self.y_train.shape[0]
                print(f'Training accuracy: {acc}')

    def single_classification(self, test_index):
        self.initialize_model()
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        self.model.eval()
        test_X = self.X[test_index]
        test_X = test_X[None, :]
        y_pred = torch.argmax(self.model.forward(test_X))
        y_true = torch.argmax(self.y[test_index])
        print(f'Predicted: {y_pred}, true: {y_true}')

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
        for i in tqdm(range(len(self.y))):
            # print(f'Trial {i+1}/{len(self.y)}...')
            self.model = EEGNet_Model(n_channels=self.n_channels, n_samples=self.n_samples, n_classes=self.n_classes).to(self.device)
            train_indices = [j for j in range(len(self.y)) if j != i]
            train_dataloader, self.X_train, self.y_train = self.create_dataloader(self.X[train_indices], self.y[train_indices])
            
            self.train(train_dataloader, n_epochs=15)

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
            ConfusionMatrixDisplay.from_predictions(y_trues.cpu(), y_preds.cpu(), display_labels=self.labels)
            plt.show()
        return accuracy