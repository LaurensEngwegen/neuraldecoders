import os
from tqdm import tqdm
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class STMF_Classifier():
    def __init__(self, X, y, labelsdict, time_window_start, time_window_stop):
        self.X = X
        self.y = y
        self.labels = []
        for _, label in labelsdict.items():
            self.labels.append(label)
        self.id2label = labelsdict
        self.time_window_start = time_window_start
        self.time_window_stop = time_window_stop
        
    def train(self, X_train, y_train):
        mean_HFB_perclass = dict()
        unique_labels = np.unique(y_train)
        for label in unique_labels:
            signal = X_train[y_train==label]
            mean_HFB_perclass[label] = np.mean(signal, 0)
        self.mean_HFB_perclass = mean_HFB_perclass

    def predict(self, X_test):
        highest_cor = -1000
        for key, mean_signal in self.mean_HFB_perclass.items():
            corr = np.corrcoef(mean_signal.flatten(), X_test.flatten())[0,1]
            if corr > highest_cor:
                highest_cor = corr
                y_pred = key
        return y_pred

    def LOO_classification(self, plot_cm=True, plot_HFB=True):
        y_preds, y_trues = [], []
        correct = 0
        total_HFB_perclass = {label: None for label in np.unique(self.y)}
        for i in tqdm(range(len(self.y))):
            train_indices = [j for j in range(len(self.y)) if j != i]
            self.train(self.X[train_indices], self.y[train_indices])
            y_pred = self.predict(self.X[i])
            y_true = self.y[i]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            # print(f"{i}. True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
            # Sum mean HFB for each classification to be able to plot
            # mean HFB over all classifications
            if i==0:
                for label in total_HFB_perclass:
                    total_HFB_perclass[label] = self.mean_HFB_perclass[label]
            else:
                for label in total_HFB_perclass:
                    total_HFB_perclass[label] += self.mean_HFB_perclass[label]
        accuracy = correct/len(self.y)
        print(f'Accuracy = {accuracy}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        for label in total_HFB_perclass:
            total_HFB_perclass[label] /= len(self.y)
        if plot_HFB:
            self.plot_HFB_perclass(total_HFB_perclass)
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy, y_trues, y_preds

    def single_classification(self, test_index):
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        y_pred = self.predict(self.X[test_index])
        y_true = self.y[test_index]
        print(f'Predicted: {y_pred}, true: {y_true}')
        self.plot_HFB_perclass(self.mean_HFB_perclass)

    def plot_HFB_perclass(self, HFB_perclass, zscore=True):
        # Check if it possible to plot (i.e. only one averaged frequency band)
        for label, hfb in HFB_perclass.items():
            if hfb.shape[1] == 1:
                HFB_perclass[label] = np.squeeze(hfb)
            else:
                return
        # Z-score each electrode for the HFB of each class
        if zscore:
            for label, hfb in HFB_perclass.items():
                for i, electrode in enumerate(hfb):
                    hfb[i] = (electrode - np.mean(electrode)) / np.std(electrode)
        n_electrodes = HFB_perclass[1].shape[0]
        # Plot Z-scored HFBs
        fig, ax = plt.subplots(1, 5, figsize=(16,4))
        plt.suptitle('Mean gamma band per class')
        for i, label in enumerate(HFB_perclass):
            ax[i].matshow(HFB_perclass[label], aspect='auto')
            ax[i].set_title(f'{self.id2label[label]}')
            # Set correct x-ticks (here: -0.5, 0, 0.5 with 0=VOT)
            start, end = ax[i].get_xlim()
            ax[i].xaxis.set_ticks((start, start+((end-start)/2), end), (self.time_window_start,0,self.time_window_stop))
            ax[i].xaxis.set_ticks_position('bottom')
            # Set correct y-ticks (electrodes)
            ax[i].yaxis.set_ticks([i for i in range(9,n_electrodes,10)], [i for i in range(10,n_electrodes+1,10)])
        ax[2].set_xlabel('Time in seconds (0 at VOT)')
        ax[0].set_ylabel('Electrodes')
        fig.tight_layout()
        plt.show()
