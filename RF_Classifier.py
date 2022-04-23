import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

class RF_Classifier():
    def __init__(self, X, y, labelsdict, print_progress=False):
        # Need to flatten for RF (i.e. concatenate all electrodes)
        self.X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        self.y = y
        self.labels = []
        for _, label in labelsdict.items():
            self.labels.append(label)
        self.id2label = labelsdict
        self.print_progress = print_progress

    def train(self, X_train, y_train):
        self.rf = RandomForestClassifier()
        self.rf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.rf.predict([X_test])[0]

    def single_classification(self, test_index):
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        y_pred = self.predict(self.X[test_index])
        y_true = self.y[test_index]
        print(f'Predicted: {y_pred}, true: {y_true}')

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
        for i in range(len(self.y)):
            if self.print_progress and (i+1) % 10 == 0:
                print(f'Trial {i+1}/{len(self.y)}...')
            train_indices = [j for j in range(len(self.y)) if j != i]
            self.train(self.X[train_indices], self.y[train_indices])
            y_pred = self.predict(self.X[i])
            y_true = self.y[i]
            y_preds.append(y_pred)
            y_trues.append(y_true)
            # print(f"{i}. True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
        accuracy = correct/len(self.y)
        print(f'Accuracy = {accuracy}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy
