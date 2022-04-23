from patient_data_mapping import PatientDataMapper
from preprocessing import Preprocessor
from trials_creation import Trials_Creator
from STMF_Classifier import STMF_Classifier
from SVM_Classifier import SVM_Classifier
from RF_Classifier import RF_Classifier
from EEGNet_tf_Classifier import EEGNet_tf_Classifier
from EEGNet_torch_Classifier import EEGNet_torch_Classifier

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Increase font size in plots
plt.rcParams.update({'font.size': 14})

# Function to preprocess data and create trials
def create_data(patient_data,
                sampling_rate,
                buffer, 
                preprocessing_type, 
                trial_window_start,
                trial_window_stop,
                trials_path=None,
                create_trials=False):
    # Define necessary paths
    if preprocessing_type == 'given_features':
        data_filenames = f'data/{patient_data.patient}/{patient_data.patient}_ECoG_CAR-gammaPwr_features.mat'
    else:
        data_filenames = []
        for i in range(patient_data.n_files):
            data_filenames.append(f'data/{patient_data.patient}/{patient_data.patient}_RawData_{i+1}.mat')

    task_path = f'data/{patient_data.patient}/{patient_data.patient}_NEW_trial_markers.mat'

    preprocessor = Preprocessor(data_filenames,
                                patient_data = patient_data,
                                sampling_rate = sampling_rate,
                                buffer = buffer,
                                preprocessing_type = preprocessing_type)

    if create_trials:
        # Set save_path to None to not save/overwrite trials
        Trials_Creator(task_path = task_path, 
                        ecog_data = preprocessor.ecog_data,
                        valid_spectra_points = preprocessor.valid_spectra_pnts,
                        break_points = preprocessor.break_points,
                        patient_data = patient_data,
                        sampling_rate = sampling_rate,
                        save_path = trials_path,
                        time_window_start = trial_window_start,
                        time_window_stop = trial_window_stop)

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
                   plot_cm=False, 
                   test_index=None,
                   print_progress=True):
    # Classification with STMF
    if classifier == 'STMF':
        stmf_cls = STMF_Classifier(X, y, labelsdict, trial_window_start, trial_window_stop)
        if test_index is not None:
            stmf_cls.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = stmf_cls.LOO_classification(plot_cm)

    # Classification with a support vector machine
    elif classifier == 'SVM':
        svm = SVM_Classifier(X, y, labelsdict, print_progress=print_progress)
        if test_index is not None:
            svm.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = svm.LOO_classification(plot_cm)

    # Classification with a random forest
    elif classifier == 'RF':
        rf = RF_Classifier(X, y, labelsdict, print_progress=print_progress)
        if test_index is not None:
            rf.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = rf.LOO_classification(plot_cm)

    # Classification with EEGNet in TensorFlow
    elif classifier == 'EEGNet':
        # Hyperparameters of EEGNet
        eegnet_kwargs = {
            'n_samples': int((trial_window_stop-trial_window_start) * sampling_rate),
            'dropoutRate': 0.5,
            'kernLength': int(sampling_rate/2),
            'F1': 4, # 8
            'D': 2, # 2
            'F2': 8, # 16
            'norm_rate': 0.25, 
            'dropoutType': 'Dropout'
        }
        eegnet_tf = EEGNet_tf_Classifier(X, y, labelsdict, n_channels=X.shape[1], print_progress=print_progress, **eegnet_kwargs)
        # print(eegnet_tf.initialize_model().summary())
        if test_index is not None:
            eegnet_tf.single_classification(test_index)
            accuracy = -1
        if LOO:
            accuracy = eegnet_tf.LOO_classification(plot_cm)

    # Classification with EEGNet in PyTorch
    # eegnet_torch = EEGNet_torch_Classifier(X, y, labels)
    # eegnet_torch.initialize_model()
    # print(eegnet_torch.model)
    # if test_index is not None:
    #     eegnet_torch.single_classification(i)
    # eegnet_torch.LOO_classification()

    return accuracy

def classification_loop(patient_IDs,
                        preprocessing_types,
                        classifiers,
                        n_experiments,
                        sampling_rate,
                        trial_window_start,
                        trial_window_stop):
    # Create dict to store accuracies 
    # (in lists for possibility to average over multiple experiments)
    accuracies = dict()
    for pid in patient_IDs:
        for pr_type in preprocessing_types:
            for clf in classifiers:
                accuracies[f'{pid}_{pr_type}_{clf}'] = []

    for pID in patient_IDs:
        patient_data = PatientDataMapper(pID)
        for preprocessing_type in preprocessing_types:
            trials_name = preprocessing_type # preprocessing_type or 'test'
            # Path where trials should be/are stored
            trials_path = f'data/{patient_data.patient}/{patient_data.patient}_{trials_name}_trials.pkl'
            # Define mapping from indices to labels (differs per patient)
            labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
            # Load (preprocessed) trials for specific patient
            X, y = load_trials(trials_path)
            # Start classification loop
            for classifier in classifiers:
                for i in range(n_experiments):
                    print(f'\nRepetition {i+1} using \'{preprocessing_type}\' trials from patient \'{pID}\'')
                    print(f'Classification with {classifier}...')
                    accuracy = classification(classifier,
                                              X, 
                                              y, 
                                              labelsdict, 
                                              sampling_rate,
                                              trial_window_start,
                                              trial_window_stop, 
                                              LOO=True,
                                              plot_cm=False, 
                                              test_index=None)
                    accuracies[f'{pID}_{preprocessing_type}_{classifier}'].append(accuracy)
    return accuracies


if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Wherer does trial window start relative to VOT (seconds)
    trial_window_start = -0.5
    # Wherer does trial window stop relative to VOT (seconds)
    trial_window_stop = 0.5
    # Spoken phonemes
    labels = ['/p/', '/oe/', '/a/', '/k/', 'Rest']
    
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6']
    # Type of preprocessing/features to extract
    preprocessing_types = ['delta', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma', 'allbands']
    # preprocessing_types = ['highgamma']
    # Define which classifiers to experiment with: 'STMF' / 'SVM' / 'RF' / 'EEGNet'
    classifiers = ['STMF']
    
    
    for pID in patient_IDs:
        patient_data = PatientDataMapper(pID)
        for ptype in preprocessing_types:
            trials_path = f'data/{patient_data.patient}/{patient_data.patient}_{ptype}_trials.pkl'
            create_data(patient_data, 
                        sampling_rate,
                        buffer, 
                        ptype, 
                        trial_window_start,
                        trial_window_stop,
                        trials_path=trials_path,
                        create_trials=True)
    

    # Number of experiments to average accuracy over 
    # (only useful for non-deterministic classifiers)
    n_experiments = 1
    '''
    accuracies = classification_loop(patient_IDs, 
                                     preprocessing_types,
                                     classifiers,
                                     n_experiments,
                                     sampling_rate,
                                     trial_window_start,
                                     trial_window_stop)

    print(f'\n\nAverage accuracies (over {n_experiments} runs)')
    for key in accuracies:
        accs = np.array(accuracies[key])
        # avg_acc = sum(accuracies[key])/len(accuracies[key])
        print(f'{key}:\t{np.mean(accs)} ({np.std(accs)})')

    '''

    # TODO:
    # - Fix the fact that raw/CAR data is 2D and features data is 3D (PROBABLY FIXED AS IN SVM_Classifier)
    # - Find out what to do with 'buffer' (maybe just leave it as is if we're gonna downsample to 512Hz anyway)
    # - Implement way to only use subset of electrodes for CAR (for acute3 or 4)

    # - Figure out how to deal with interpretation of results: rest class might be 100% accurate and influence total accuracy
    #       Maybe use F1 score, or visualize confusion matrix, but that's not possible to do for all experiments


    # Most important parameters for random forest:
    # - n_estimators (100)
    # - max_depth (None) --> over-/underfitting
    # - min_samples_split (2) --> over-/underfitting
    # - min_samples_leaf (1)
    # - max_leaf_nodes (None)


    # Voor Acute004: specifieke CAR channels zijn defined in datafile
    # deze zouden in theorie (bijna) alleen maar noise moeten bevatten
    # en zijn daarom goed/optimaal voor CAR

    # Voor Acute 3 of 4(?): speech signal zit in data