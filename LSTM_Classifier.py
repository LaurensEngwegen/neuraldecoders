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

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        output = self.linear(hidden_state[-1]) # Last layer, last state
        return output

class LSTM_Classifier():
    def __init__(self, x, y, labelsdict, 
                 n_hidden=256, 
                 n_layers=2, 
                 dropout_rate=0.5,
                 loss=nn.CrossEntropyLoss,
                 optimizer=optim.Adam)
        
        n_features = x.shape[0] # nr electrodes
        n_classes = len(np.unique(y)) # nr classes

        self.model = LSTMModel(n_features, n_classes, n_hidden, n_layers, dropout_rate)
        self.loss = loss()
        self.optimizer = optimizer(self.model.parameters())
