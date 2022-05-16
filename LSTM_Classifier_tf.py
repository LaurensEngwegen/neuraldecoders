from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils as np_utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


class LSTM_Classifier_tf():
    def __init__(self, X, y, labelsdict,
                 n_channels = 60,
                 n_samples = 512,):
        self.X = X
        self.y = y
        self.labels = []
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
            self.y = np.where(self.y==id, i, self.y)
        self.y = np_utils.to_categorical(self.y)
        self.id2label = labelsdict
        self.n_classes = len(self.labels)

        # Variables and parameters needed for model
        self.n_channels = n_channels
        self.n_samples = n_samples

    def initialize_model(self):
        # input   = layers.Input(shape = (self.n_channels, self.n_samples))
        # out = layers.LSTM(32)(input)
        # # out = layers.LSTM(32)(out)
        # out = layers.Dense(self.n_classes)(out)
        # out = layers.Activation('softmax')(out)

        # model = Model(inputs=input, outputs=out)
        model = models.Sequential([
            layers.Input(shape = (self.n_channels, self.n_samples)),
            # layers.LSTM(256, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(self.n_classes),
            layers.Activation('softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        return model


    def train(self, X_train, y_train, epochs=200):
        X_val = X_train[-5:]
        X_train = X_train[:-5]
        y_val = y_train[-5:]
        y_train = y_train[:-5]
        self.fitted_model = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = 25, epochs = epochs, verbose = 1)

    def predict(self, X_test):
        X_test = X_test[np.newaxis, ...]
        y_pred = self.model.predict(X_test)
        return y_pred

    def single_classification(self, test_index):
        self.model = self.initialize_model()
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        y_pred = np.argmax(self.predict(self.X[test_index]))
        y_true = np.argmax(self.y[test_index])
        print(f'Predicted: {y_pred}, true: {y_true}')

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
        # self.model = self.initialize_model()
        # print(self.X[1:2])
        # print(self.model(self.X[1:2]))
        # for i in range(15):
        #     self.single_classification(i)

        self.model = self.initialize_model()
        print(self.model.summary())
        for i in tqdm(range(len(self.y))):
            self.model = self.initialize_model()
            train_indices = [j for j in range(len(self.y)) if j != i]
            self.train(self.X[train_indices], self.y[train_indices])
            y_pred = np.argmax(self.predict(self.X[i]))
            y_true = np.argmax(self.y[i])
            y_preds.append(y_pred)
            y_trues.append(y_true)
            print(f"{i}. True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
            # K.clear_session()
        accuracy = correct/len(self.y)
        print(f'Accuracy = {accuracy}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy, y_trues, y_preds