from matplotlib.axis import XAxis
from patient_data_mapping import PatientDataMapper
from preprocessing import Preprocessor
from trials_creation import Trials_Creator
from STMF_Classifier import STMF_Classifier
from SVM_Classifier import SVM_Classifier
from kNN_Classifier import kNN_Classifier
from FFN_Classifier import FFN_Classifier
from EEGNet_tf_Classifier import EEGNet_tf_Classifier
from EEGNet_torch_Classifier import EEGNet_torch_Classifier
from LSTM_Classifier import LSTM_Classifier
from LSTM_Classifier_tf import LSTM_Classifier_tf

import os
import numpy as np
from sklearn.metrics import accuracy_score
import pickle as pkl
import matplotlib.pyplot as plt

# Increase font size in plots
plt.rcParams.update({'font.size': 16})

# Function to preprocess data and create trials
def preprocessing(patient_data,
                  sampling_rate,
                  buffer, 
                  preprocessing_type,
                  task='phonemes'):
    # Define necessary paths
    if preprocessing_type == 'given_features':
        data_filenames = f'data/{task}/{patient_data.patient}/{patient_data.patient}_ECoG_CAR-gammaPwr_features.mat'
    else:
        data_filenames = []
        if task == 'phonemes':
            for i in range(patient_data.n_files):
                data_filenames.append(f'data/{task}/{patient_data.patient}/{patient_data.patient}_RawData_{i+1}.mat')
        else:
            for i in range(patient_data.n_files):
                data_filenames.append(f'data/{task}/{patient_data.patient}/{patient_data.patient}_run_{i+1}.mat')

    preprocessor = Preprocessor(data_filenames,
                                patient_data = patient_data,
                                sampling_rate = sampling_rate,
                                buffer = buffer,
                                preprocessing_type = preprocessing_type)
    
    return preprocessor.ecog_data, preprocessor.valid_spectra_pnts, preprocessor.break_points

def trials_creation(patient_data,
                    ecog_data,
                    valid_spectra_pnts,
                    break_points,
                    sampling_rate,
                    trials_path,
                    trial_window_start,
                    trial_window_stop,
                    restvsactive,
                    task='phonemes'):
        if task == 'phonemes':
            task_path = [f'data/{task}/{patient_data.patient}/{patient_data.patient}_NEW_trial_markers.mat']
        else:
            task_path = []
            for i in range(patient_data.n_files):
                task_path.append(f'data/{task}/{patient_data.patient}/{patient_data.patient}_run_{i+1}.mat')
        # Set save_path to None to not save/overwrite trials
        Trials_Creator(task_path = task_path, 
                        ecog_data = ecog_data,
                        valid_spectra_points = valid_spectra_pnts,
                        break_points = break_points,
                        patient_data = patient_data,
                        sampling_rate = sampling_rate,
                        save_path = trials_path,
                        time_window_start = trial_window_start,
                        time_window_stop = trial_window_stop,
                        restvsactive = restvsactive)

# Function to load trials
def load_trials(trials_path):
    if trials_path is not None:
        with open(trials_path, 'rb') as f:
            X, y = pkl.load(f)
        return X, y
    else:
        print(f'Error: no trials saved')
        return None

def classification(classifier, 
                   X, 
                   y, 
                   labelsdict, 
                   sampling_rate,
                   trial_window_start, 
                   trial_window_stop, 
                   LOO=False,
                   make_plots=False, 
                   test_index=None):
    # Classification with STMF
    if classifier == 'STMF':
        stmf_cls = STMF_Classifier(X, y, labelsdict, trial_window_start, trial_window_stop)
        if test_index is not None:
            stmf_cls.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = stmf_cls.LOO_classification(plot_cm=make_plots, plot_HFB=make_plots)

    # Classification with a support vector machine
    elif classifier == 'SVM':
        svm = SVM_Classifier(X, y, labelsdict)
        if test_index is not None:
            svm.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = svm.LOO_classification(make_plots)

    # Classification on basis of k nearest neighbours
    elif classifier in ['kNN3','kNN5','kNN7','kNN9','kNN11','kNN13','kNN15','kNN17','kNN19']:
        if len(classifier) == 4:
            n_neighbors = int(classifier[len(classifier)-1])
        else:
            n_neighbors = int(classifier[len(classifier)-2:])
        kNN = kNN_Classifier(X, y, labelsdict, n_neighbors)
        if test_index is not None:
            kNN.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = kNN.LOO_classification(make_plots)

    # Classification with (simple) feedforward neural network
    elif classifier in ['FFN256-128','FFN128-64','FFN64-32','FFN32-16','FFN16-8']:
        if classifier == 'FFN256-128':
            n_nodes = [256,128]
        elif classifier == 'FFN128-64':
            n_nodes = [128,64]
        elif classifier == 'FFN64-32':
            n_nodes = [64,32]
        elif classifier == 'FFN32-16':
            n_nodes = [32,16]
        elif classifier == 'FFN16-8':
            n_nodes = [16,8]
        ffn = FFN_Classifier(X, y, labelsdict, n_nodes)
        if test_index is not None:
            ffn.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = ffn.LOO_classification(make_plots)

    # Classification with EEGNet in TensorFlow
    elif classifier in ['EEGNet', 'EEGNet4-2', 'EEGNet16-2']:
        # Hyperparameters of EEGNet
        eegnet_kwargs = {
            'n_samples': int((trial_window_stop-trial_window_start) * sampling_rate),
            'dropoutRate': 0.5,
            'kernLength': int(sampling_rate/2),
            'norm_rate': 0.25, 
            'dropoutType': 'Dropout'
        }
        # Default EEGNet8-2
        if classifier == 'EEGNet':
            eegnet_kwargs['F1'] = 8
            eegnet_kwargs['D'] = 2
            eegnet_kwargs['F2'] = 16
        elif classifier == 'EEGNet4-2':
            eegnet_kwargs['F1'] = 4
            eegnet_kwargs['D'] = 2
            eegnet_kwargs['F2'] = 8
        else: # EEGNet16-2
            eegnet_kwargs['F1'] = 16
            eegnet_kwargs['D'] = 2
            eegnet_kwargs['F2'] = 32
        eegnet_tf = EEGNet_tf_Classifier(X, y, labelsdict, n_channels=X.shape[1], **eegnet_kwargs)
        # print(eegnet_tf.initialize_model().summary())
        if test_index is not None:
            eegnet_tf.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = eegnet_tf.LOO_classification(make_plots)

    # Classification with EEGNet in PyTorch
    elif classifier == 'EEGNet_torch':
        # Hyperparameters of EEGNet
        eegnet_kwargs = {
            'n_samples': int((trial_window_stop-trial_window_start) * sampling_rate),
            'dropoutRate': 0.5,
            'kernLength': int(sampling_rate/2),
            'F1': 8,
            'D': 2, 
            'F2': 16,
            'norm1': 1.0,
            'norm2': 0.25
        }
        eegnet = EEGNet_torch_Classifier(X, y, labelsdict, n_channels=X.shape[1], **eegnet_kwargs)
        if test_index is not None:
            eegnet.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = eegnet.LOO_classification(make_plots)

    # Classification wit (stacked) LSTM
    elif classifier in ['LSTM256', 'LSTM128', 'LSTM64', 'LSTM32', 'LSTM16']:
        if classifier == 'LSTM256':
            n_hidden = 256
        elif classifier == 'LSTM128':
            n_hidden = 128
        elif classifier == 'LSTM64':
            n_hidden = 64
        elif classifier == 'LSTM32':
            n_hidden = 32
        elif classifier == 'LSTM16':
            n_hidden = 16

        lstm = LSTM_Classifier(X, y, labelsdict, n_hidden=n_hidden)
        if test_index is not None:
            lstm.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = lstm.LOO_classification(make_plots)
    
    elif classifier == 'LSTM_tf':
        lstm = LSTM_Classifier_tf(X, y, labelsdict)
        if test_index is not None:
            lstm.single_classification(test_index)
            accuracy, y_true, y_pred = -1, -1, -1
        if LOO:
            accuracy, y_true, y_pred = lstm.LOO_classification(make_plots)
    
    return accuracy, y_true, y_pred

def classification_loop(patient_IDs,
                        preprocessing_types,
                        classifiers,
                        classification_type,
                        labels,
                        n_experiments,
                        sampling_rate,
                        trial_window_start,
                        trial_window_stop,
                        make_plots,
                        save_results,
                        LOO=True,
                        test_index=None,
                        task='phonemes'):
    # Create dict to store accuracies 
    # (in lists for possibility to average over multiple experiments)
    results = dict()
    # Start classification loop
    for classifier in classifiers:
        results[classifier] = dict()
        for preprocessing_type in preprocessing_types:
            results[classifier][preprocessing_type] = dict()
            for pID in patient_IDs:
                patient_data = PatientDataMapper(pID, task)
                # Path where trials are stored
                trials_path = f'data/{task}/{patient_data.patient}/{patient_data.patient}_{preprocessing_type}{classification_type}_trials.pkl'
                # Define mapping from indices to labels (differs per patient)
                labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
                # Load (preprocessed) trials for specific patient
                X, y = load_trials(trials_path)
                results[classifier][preprocessing_type][patient_data.patient] = {'y_true': [], 'y_pred': []}
                for i in range(n_experiments):
                    print(f'\nRepetition {i+1} using \'{preprocessing_type}\' trials from patient \'{patient_data.patient}\' ({X.shape[1]} channels, {X.shape[0]} trials)')
                    print(f'Classification with {classifier}...')
                    accuracy, y_true, y_pred = classification(classifier,
                                                              X, 
                                                              y, 
                                                              labelsdict, 
                                                              sampling_rate,
                                                              trial_window_start,
                                                              trial_window_stop, 
                                                              make_plots=make_plots, 
                                                              LOO=LOO,
                                                              test_index=test_index)
                    results[classifier][preprocessing_type][patient_data.patient]['y_true'].append(y_true)
                    results[classifier][preprocessing_type][patient_data.patient]['y_pred'].append(y_pred)
                if save_results:
                    directory = f'results/{task}/{classifier}/{preprocessing_type}{classification_type}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    file = f'{directory}/{patient_data.patient}_results.pkl'
                    with open(file, 'wb+') as f:
                        pkl.dump(results[classifier][preprocessing_type][patient_data.patient], f)
                    print(f'Results stored in \'{file}\'')
    return results

def print_results(results, n_experiments):
    # Accuracies is dictionary with [classifier][patient][preprocessing_type]
    print(f'\n\nAverage accuracies (over {n_experiments} runs)')
    for classifier, processing_dict in results.items():
        print(f'\n{classifier}')
        for processing, patient_dict in processing_dict.items():
            for patient, y_dict in patient_dict.items():
                all_accs = []
                for i in range(n_experiments):
                    all_accs.append(accuracy_score(y_dict['y_true'][i], y_dict['y_pred'][i]))
                all_accs = np.array(all_accs)
                print(f'{patient}-{processing}:\t{np.mean(all_accs)} ({np.std(all_accs)})')

def get_accuracy(file):
    with open(file, 'rb') as f:
        results = pkl.load(f)
    # Previously stored results as list of accuracies instead of y's
    if isinstance(results, list):
        results = np.array(results)
        return np.mean(results), np.std(results)
    else:
        accs = []
        n_experiments = len(results['y_true'])
        for i in range(n_experiments):
            accs.append(accuracy_score(results['y_true'][i], results['y_pred'][i]))
        accs = np.array(accs)
        return np.mean(accs), np.std(accs)



def plot_features_results(classifiers, preprocessing_types, patient_IDs, restvsactive=False, task='phonemes'):
    if restvsactive:
        preprocessing_types = [ptype+'_RvA' for ptype in preprocessing_types]
    # Define plot variables
    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID, task).patient)
    x_axis = np.arange(len(patient_IDs))
    width = 0.1
    # pos = [-.25, -.15, -.05, .05, .15, .25]
    pos = [-.3, -.2, -.1, 0, .1, .2, .35]
    # Read all result data
    for classifier in classifiers:
        all_data = []
        for preprocessing_type in preprocessing_types:
            data = []
            directory = f'results/{task}/{classifier}/{preprocessing_type}'
            for patient_ID in patient_IDs:
                patient_data = PatientDataMapper(patient_ID, task)
                file = f'{directory}/{patient_data.patient}_results.pkl'
                acc, _ = get_accuracy(file)
                data.append(acc)
            all_data.append(data) # Patients x Preprocessing_types
        # Barplot
        plt.figure(figsize=(12,8))
        for i, data in enumerate(all_data):
            if restvsactive:
                plt.bar(x_axis + pos[i], data, width=width, label=preprocessing_types[i][:-4])
            else:
                plt.bar(x_axis + pos[i], data, width=width, label=preprocessing_types[i])
        plot_article_acc(x_axis, pos[len(pos)-1], width, restvsactive, task)
        if restvsactive:
            title = f'Accuracy of {classifier} on active vs. rest using different frequency bands'
        else:
            title = f'Accuracy of {classifier} on 5-class classification using different frequency bands'
        plt.title(title)
        plt.xticks(x_axis, patient_labels)
        plt.yticks(np.arange(0,1.01,0.1))
        plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
        plt.grid(alpha=0.35)
        plt.tight_layout()
        plt.show()

def plot_clf_optimization(classifiers, preprocessing_type, patient_IDs, restvsactive=False, task='phonemes'):
    for classifier in classifiers:
        # k's tested for kNN
        if classifier == 'kNN':
            params = ['3','5','7','9','11','13','15','17','19']
            labelprefix = '$k$ = '
        # Architectures tested for FFN
        elif classifier == 'FFN':
            params = ['256-128','128-64','64-32','32-16']
            labelprefix = 'FFN'
        n_params = len(params)
        tested_classifiers = []
        for param in params:
            tested_classifiers.append(f'{classifier}{param}')

        if restvsactive:
            preprocessing_type = f'{preprocessing_type}_RvA'
        # Define plot variables
        patient_labels = []
        for patient_ID in patient_IDs:
            patient_labels.append(PatientDataMapper(patient_ID, task).patient)
        x_axis = np.arange(len(patient_IDs))
        width = 0.08
        pos = []
        # Make 'pos' a list of len=n_params with values around 0
        if n_params % 2 == 0: # even number of params
            for i in np.arange(-0.5*width-((n_params-2)/2)*width, 0, width):
                pos.append(round(i,2))
            for i in np.arange(0.5*width, 0.5*width+((n_params-1)/2)*width, width):
                pos.append(round(i,2))
        else: # odd number of params
            for i in np.arange(0-((n_params-1)/2)*width, 0, width):
                pos.append(round(i,2))
            for i in np.arange(0, ((n_params+1)/2)*width, width):
                pos.append(round(i,2))
        
        # Read all result data
        all_data = []
        for tested_clf in tested_classifiers:
            data = []
            directory = f'results/{task}/{tested_clf}/{preprocessing_type}'
            for patient_ID in patient_IDs:
                patient_data = PatientDataMapper(patient_ID, task)
                file = f'{directory}/{patient_data.patient}_results.pkl'
                acc, _ = get_accuracy(file)
                data.append(acc)
            all_data.append(data) # Patients x Preprocessing_types
        # Barplot
        plt.figure(figsize=(12,8))
        for i, data in enumerate(all_data):
            plt.bar(x_axis + pos[i], data, width=width, label=f'{labelprefix}{params[i]}')
        if restvsactive:
            title = f'Accuracy of {classifier} on active vs. rest'
        else:
            title = f'Accuracy of {classifier} on 5-class classification'
        plt.title(title)
        plt.xticks(x_axis, patient_labels)
        plt.yticks(np.arange(0,1.01,0.1))
        plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
        plt.grid(alpha=0.35)
        plt.tight_layout()
        plt.show()
    
def plot_classifier_results(patient_IDs, task='phonemes'):
    result_info = {
        'STMF': {'ptype': 'gamma', 'title': 'STMF'},
        'SVM': {'ptype': 'gamma', 'title': 'SVM'},
        'kNN11': {'ptype': 'gamma', 'title': 'kNN (k=11)'},
        'FFN128-64': {'ptype': 'gamma', 'title': 'FFN (gamma band)'},
        'EEGNet': {'ptype': 'CAR', 'title': 'EEGNet'}
    }
    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID, task).patient)
    x_axis = np.arange(len(patient_IDs))
    # pos = [-0.3, -0.15, -0.05, 0.05, 0.15, 0.25]
    pos = [-0.2, -0.1, 0, 0.1, 0.2]
    width = 0.1

    all_accs, all_stds, titles = [], [], []
    for classifier in result_info:
        accs, stds = [], []
        preprocessing_type = result_info[classifier]['ptype']
        directory = f'results/{task}/{classifier}/{preprocessing_type}'
        for patient in patient_labels:
            file = f'{directory}/{patient}_results.pkl'
            acc, std = get_accuracy(file)
            accs.append(acc)
            stds.append(std)
        all_accs.append(accs)
        all_stds.append(stds)
        titles.append(result_info[classifier]['title'])

    plt.figure(figsize=(12,8))
    # Add entry to 'pos' to plot article acc.
    # plot_article_acc(x_axis, pos[0], width, task)
    for i, accuracies in enumerate(all_accs):
        plt.bar(x_axis+pos[i], accuracies, yerr=all_stds[i], width=width, label=titles[i])
    plt.title(f'Accuracy of different classifiers')
    plt.xticks(x_axis, patient_labels)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.show()

# TODO: add patient 4 acc (0.685) when data i
def plot_article_acc(x_axis, pos, width, restvsactive=False, task='phonemes'):
    if task == 'phonemes':
        article_accuracies = [0.814, 0.704, 0.831, 0.741, 0, 0, 0]
        if restvsactive:
            article_accuracies = [0.831, 1, 1, 0.667, 0, 0, 0]
        label = 'Article STMF'
    else:
        return
    plt.bar(x_axis+pos, article_accuracies, width=width, label=label, color='lightgrey')
