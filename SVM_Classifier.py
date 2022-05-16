import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


class SVM_Classifier():
    def __init__(self, X, y, labelsdict, kernel='linear'):
        # self.X = X
        X = X.squeeze()
        X_second_dim = 1
        for i in range(1, len(X.shape)):
            X_second_dim *= X.shape[i]
        self.X = X.reshape(X.shape[0], X_second_dim)
        self.y = y
        self.labels = []
        for _, label in labelsdict.items():
            self.labels.append(label)
        self.id2label = labelsdict
        self.kernel = kernel

    def train(self, X_train, y_train):
        self.clf = SVC(kernel=self.kernel)
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict([X_test])[0]

    def single_classification(self, test_index):
        train_indices = [j for j in range(len(self.y)) if j != test_index]
        self.train(self.X[train_indices], self.y[train_indices])
        y_pred = self.predict(self.X[test_index])
        y_true = self.y[test_index]
        print(f'Predicted: {y_pred}, true: {y_true}')

    def LOO_classification(self, plot_cm=True):
        y_preds, y_trues = [], []
        correct = 0
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
        accuracy = correct/len(self.y)
        print(f'Accuracy = {accuracy}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy, y_trues, y_preds
