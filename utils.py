from patient_data_mapping import PatientDataMapper
from preprocessing import Preprocessor
from trials_creation import Trials_Creator
from STMF_Classifier import STMF_Classifier
from SVM_Classifier import SVM_Classifier
from kNN_Classifier import kNN_Classifier
from RF_Classifier import RF_Classifier
from EEGNet_tf_Classifier import EEGNet_tf_Classifier
from EEGNet_torch_Classifier import EEGNet_torch_Classifier
from LSTM_Classifier import LSTM_Classifier
from LSTM_Classifier_tf import LSTM_Classifier_tf

import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# Increase font size in plots
plt.rcParams.update({'font.size': 14})

# Function to preprocess data and create trials
def preprocessing(patient_data,
                  sampling_rate,
                  buffer, 
                  preprocessing_type):
    # Define necessary paths
    if preprocessing_type == 'given_features':
        data_filenames = f'data/{patient_data.patient}/{patient_data.patient}_ECoG_CAR-gammaPwr_features.mat'
    else:
        data_filenames = []
        for i in range(patient_data.n_files):
            data_filenames.append(f'data/{patient_data.patient}/{patient_data.patient}_RawData_{i+1}.mat')

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
                    restvsactive):

        task_path = f'data/{patient_data.patient}/{patient_data.patient}_NEW_trial_markers.mat'
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
            accuracy = -1
        if LOO:
            accuracy = stmf_cls.LOO_classification(plot_cm=make_plots, plot_HFB=make_plots)

    # Classification with a support vector machine
    elif classifier == 'SVM':
        svm = SVM_Classifier(X, y, labelsdict)
        if test_index is not None:
            svm.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = svm.LOO_classification(make_plots)

    # Classification on basis of k nearest neighbours
    elif classifier == 'kNN':
        kNN = kNN_Classifier(X, y, labelsdict)
        if test_index is not None:
            kNN.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = kNN.LOO_classification(make_plots)

    # Classification with a random forest
    elif classifier == 'RF':
        rf = RF_Classifier(X, y, labelsdict)
        if test_index is not None:
            rf.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = rf.LOO_classification(make_plots)

    # Classification with EEGNet in TensorFlow
    elif classifier == 'EEGNet' or classifier == 'EEGNet4-2' or classifier == 'EEGNet16-2':
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
            accuracy = -1
        if LOO:
            accuracy = eegnet_tf.LOO_classification(make_plots)

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
        eegnet = EEGNet_torch_Classifier(X.astype('float32'), y, labelsdict, n_channels=X.shape[1], **eegnet_kwargs)
        if test_index is not None:
            eegnet.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = eegnet.LOO_classification(make_plots)

    # Classification wit (stacked) LSTM
    elif classifier == 'LSTM':
        lstm = LSTM_Classifier(X, y, labelsdict)
        if test_index is not None:
            lstm.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = lstm.LOO_classification(make_plots)
    
    elif classifier == 'LSTM_tf':
        lstm = LSTM_Classifier_tf(X, y, labelsdict)
        if test_index is not None:
            lstm.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = lstm.LOO_classification(make_plots)
    
    
    return accuracy

def classification_loop(patient_IDs,
                        preprocessing_types,
                        classifiers,
                        classification_type,
                        n_experiments,
                        sampling_rate,
                        trial_window_start,
                        trial_window_stop,
                        make_plots,
                        save_results,
                        LOO=True,
                        test_index=None):
    # Create dict to store accuracies 
    # (in lists for possibility to average over multiple experiments)
    accuracies = dict()
    # Start classification loop
    for classifier in classifiers:
        accuracies[classifier] = dict()
        for preprocessing_type in preprocessing_types:
            accuracies[classifier][preprocessing_type] = dict()
            for pID in patient_IDs:
                patient_data = PatientDataMapper(pID)
                # Path where trials are stored
                trials_path = f'data/{patient_data.patient}/{patient_data.patient}_{preprocessing_type}{classification_type}_trials.pkl'
                # Define mapping from indices to labels (differs per patient)
                labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
                # Load (preprocessed) trials for specific patient
                X, y = load_trials(trials_path)
                accuracies[classifier][preprocessing_type][patient_data.patient] = []
                for i in range(n_experiments):
                    print(f'\nRepetition {i+1} using \'{preprocessing_type}\' trials from patient \'{pID}\' ({X.shape[1]} channels, {X.shape[0]} trials)')
                    print(f'Classification with {classifier}...')
                    accuracy = classification(classifier,
                                              X, 
                                              y, 
                                              labelsdict, 
                                              sampling_rate,
                                              trial_window_start,
                                              trial_window_stop, 
                                              make_plots=make_plots, 
                                              LOO=LOO,
                                              test_index=test_index)
                    accuracies[classifier][preprocessing_type][patient_data.patient].append(accuracy)
                if save_results:
                    directory = f'results/{classifier}/{preprocessing_type}{classification_type}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    file = f'{directory}/{patient_data.patient}{classification_type}_results.pkl'
                    with open(file, 'wb+') as f:
                        pkl.dump(accuracies[classifier][preprocessing_type][patient_data.patient], f)
                    print(f'Results stored in \'{file}\'')
    return accuracies

def print_results(accuracies, n_experiments):
    # Accuracies is dictionary with [classifier][patient][preprocessing_type]
    print(f'\n\nAverage accuracies (over {n_experiments} runs)')
    for classifier, processing_dict in accuracies.items():
        print(f'\n{classifier}')
        for processing, patient_dict in processing_dict.items():
            for patient, all_accs in patient_dict.items():
                all_accs = np.array(all_accs)
                print(f'{patient}-{processing}:\t{np.mean(all_accs)} ({np.std(all_accs)})')

def plot_features_results(classifiers, preprocessing_types, patient_IDs):
    # Define plot variables
    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID).patient)
    x_axis = np.arange(len(patient_IDs))
    width = 0.1
    pos = [-.4, -.3, -.2, -.1, 0, .1, .2, .3, .4]
    # Read all result data
    for classifier in classifiers:
        all_data = []
        for preprocessing_type in preprocessing_types:
            data = []
            directory = f'results/{classifier}/{preprocessing_type}'
            for patient_ID in patient_IDs:
                patient_data = PatientDataMapper(patient_ID)
                file = f'{directory}/{patient_data.patient}_results.pkl'
                with open(file, 'rb') as f:
                    results = np.mean(pkl.load(f))
                data.append(results)
            all_data.append(data) # Patients x Preprocessing_types
        # Barplot
        plt.figure(figsize=(8,6))
        for i, data in enumerate(all_data):
            plt.bar(x_axis + pos[i], data, width=width, label=preprocessing_types[i])
        plt.title(f'Accuracy of {classifier}')
        plt.xticks(x_axis, patient_labels)
        plt.yticks(np.arange(0,1.01,0.1))
        plt.legend()
        plt.grid(alpha=0.35)
        plt.show()

def plot_classifier_results(patient_IDs):
    result_info = {
        'STMF': 'highgamma',
        'SVM': 'highgamma',
        'kNN': 'highgamma',
        'EEGNet': 'CAR'
    }

    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID).patient)
    x_axis = np.arange(len(patient_IDs))
    pos = [-0.3, -0.1, 0.1, 0.3]
    width = 0.2

    all_accs, all_stds = [], []
    for classifier, preprocessing_type in result_info.items():
        accs, stds = [], []
        directory = f'results/{classifier}/{preprocessing_type}'
        for patient in patient_labels:
            file = f'{directory}/{patient}_results.pkl'
            with open(file, 'rb') as f:
                results = pkl.load(f)
            acc = np.mean(results)
            std = np.std(results)
            accs.append(acc)
            stds.append(std)
        all_accs.append(accs)
        all_stds.append(stds)

    plt.figure(figsize=(8,6))
    for i, accuracies in enumerate(all_accs):
        plt.bar(x_axis+pos[i], accuracies, yerr=all_stds[i], width=width, label=list(result_info)[i])
    plt.title(f'Accuracy of different classifiers')
    plt.xticks(x_axis, patient_labels)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend()
    plt.grid(alpha=0.35)
    plt.show()
