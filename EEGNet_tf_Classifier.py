from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from tensorflow.keras import utils as np_utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

class EEGNet_tf_Classifier():
    def __init__(self, 
                 X, 
                 y, 
                 labelsdict, 
                 n_channels = 60,
                 n_samples = 512,
                 dropoutRate = 0.5,
                 kernLength = 256, # sampling_rate/2
                 F1 = 8, 
                 D = 2, 
                 F2 = 16, 
                 norm_rate = 0.25, 
                 dropoutType = 'Dropout',):
        self.X = X
        self.y = y
        self.labels = []
        for i, (id, label) in enumerate(labelsdict.items()):
            self.labels.append(label)
            self.y = np.where(y==id, i, self.y)
        self.y = np_utils.to_categorical(self.y)
        self.id2label = labelsdict
        self.n_classes = len(self.labels)

        # Variables and parameters needed for model
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropoutType = dropoutType

    def initialize_model(self):
        if self.dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif self.dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                            'or Dropout, passed as a string.')
        
        input1   = Input(shape = (self.n_channels, self.n_samples, 1))

        ##################################################################

        block1       = Conv2D(self.F1, (1, self.kernLength), padding = 'same',
                              input_shape = (self.n_channels, self.n_samples, 1),
                              use_bias = False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((self.n_channels, 1), use_bias = False, 
                                        depth_multiplier = self.D,
                                        depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = dropoutType(self.dropoutRate)(block1)
        
        block2       = SeparableConv2D(self.F2, (1, 16),
                                       use_bias = False, padding = 'same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = dropoutType(self.dropoutRate)(block2)
            
        flatten      = Flatten(name = 'flatten')(block2)
        
        dense        = Dense(self.n_classes, name = 'dense', 
                             kernel_constraint = max_norm(self.norm_rate))(flatten)
        softmax      = Activation('softmax', name = 'softmax')(dense)
        
        model = Model(inputs=input1, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        return model

    # TODO: fitted_model (History object) is probably not needed
    # TODO: create custom training loop
    def train(self, X_train, y_train):
        # X_val = X_train[-5:]
        # X_train = X_train[:-5]
        # y_val = y_train[-5:]
        # y_train = y_train[:-5]
        self.fitted_model = self.model.fit(X_train, y_train, batch_size = 5, epochs = 25, verbose = 0)

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
        for i in tqdm(range(len(self.y))):
            self.model = self.initialize_model()
            train_indices = [j for j in range(len(self.y)) if j != i]
            self.train(self.X[train_indices], self.y[train_indices])
            y_pred = np.argmax(self.predict(self.X[i]))
            y_true = np.argmax(self.y[i])
            y_preds.append(y_pred)
            y_trues.append(y_true)
            # print(f"{i}. True - predicted ==> {y_true} - {y_pred}")
            if y_pred == y_true:
                correct += 1
            # K.clear_session()
        accuracy = correct/len(self.y)
        print(f'Accuracy = {accuracy}')
        print(f'correct: {correct}, len(y): {len(self.y)}')
        if plot_cm:
            ConfusionMatrixDisplay.from_predictions(y_trues, y_preds, display_labels=self.labels)
            plt.show()
        return accuracy