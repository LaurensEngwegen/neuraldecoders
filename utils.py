from patient_data_mapping import PatientDataMapper
from preprocessing import Preprocessor
from trials_creation import Trials_Creator
from classifiers.STMF_Classifier import STMF_Classifier
from classifiers.SVM_Classifier import SVM_Classifier
from classifiers.kNN_Classifier import kNN_Classifier
from classifiers.FFN_Classifier import FFN_Classifier
from classifiers.EEGNet_tf_Classifier import EEGNet_tf_Classifier
from classifiers.EEGNet_torch_Classifier import EEGNet_torch_Classifier
from classifiers.LSTM_Classifier import LSTM_Classifier
from classifiers.LSTM_Classifier_tf import LSTM_Classifier_tf

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle as pkl
import matplotlib.pyplot as plt


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
        if task == 'phonemes' or task == 'phonemes_noRest':
            for i in range(patient_data.n_files):
                data_filenames.append(f'data/phonemes/{patient_data.patient}/{patient_data.patient}_RawData_{i+1}.mat')
        elif task == 'gestures' or task == 'small_gestures':
            for i in range(patient_data.n_files):
                data_filenames.append(f'data/gestures/{patient_data.patient}/{patient_data.patient}_run_{i+1}.mat')
        elif task == 'pretrain_phonemes':
            data_filenames.append(f'data/pretrain_phonemes/{patient_data.patient}/4P_{patient_data.patient}.mat')
        elif task == 'pretrain_gestures' or task == 'pretrain_small_gestures':
            data_filenames.append(f'data/pretrain_gestures/{patient_data.patient}/4G_{patient_data.patient}.mat')

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
        if task == 'phonemes' or task == 'phonemes_noRest':
            task_path = [f'data/phonemes/{patient_data.patient}/{patient_data.patient}_NEW_trial_markers.mat']
        elif task == 'gestures' or task == 'small_gestures':
            task_path = []
            for i in range(patient_data.n_files):
                task_path.append(f'data/gestures/{patient_data.patient}/{patient_data.patient}_run_{i+1}.mat')
        elif task == 'pretrain_phonemes':
            task_path = [f'data/pretrain_phonemes/{patient_data.patient}/4P_{patient_data.patient}.mat']
        elif task == 'pretrain_gestures' or task == 'pretrain_small_gestures':
            task_path = [f'data/pretrain_gestures/{patient_data.patient}/4G_{patient_data.patient}.mat']
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
                        restvsactive = restvsactive,
                        task = task)

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
    # Intermediate results during training of neural networks    
    intermediate_y_true, intermediate_y_pred = None, None

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
    elif classifier in ['kNN1','kNN3','kNN5','kNN7','kNN9','kNN11','kNN13','kNN15','kNN17','kNN19']:
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
            accuracy, y_true, y_pred, intermediate_y_true, intermediate_y_pred = ffn.LOO_classification(make_plots)

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
            accuracy, y_true, y_pred, intermediate_y_true, intermediate_y_pred = eegnet_tf.LOO_classification(make_plots)

    # Classification with (stacked) LSTM
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
            accuracy, y_true, y_pred, _, _ = lstm.LOO_classification(make_plots)
    
    return accuracy, y_true, y_pred, intermediate_y_true, intermediate_y_pred

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
                        save_intermediate_results,
                        LOO=True,
                        test_index=None,
                        task='phonemes'):
    # Create dict to store accuracies 
    # (in lists for possibility to average over multiple experiments)
    results = dict()
    intermediate_results = dict()
    # Start classification loop
    for classifier in classifiers:
        results[classifier] = dict()
        intermediate_results[classifier] = dict()
        for preprocessing_type in preprocessing_types:
            results[classifier][preprocessing_type] = dict()
            intermediate_results[classifier][preprocessing_type] = dict()
            for pID in patient_IDs:
                patient_data = PatientDataMapper(pID, task)
                # Path where trials are stored
                trials_path = f'data/{task}/{patient_data.trials_fileID}/{patient_data.trials_fileID}_{preprocessing_type}{classification_type}_trials.pkl'
                # Define mapping from indices to labels (differs per patient)
                labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
                # Load (preprocessed) trials for specific patient
                X, y = load_trials(trials_path)
                results[classifier][preprocessing_type][patient_data.patient] = {'y_true': [], 'y_pred': []}
                intermediate_results[classifier][preprocessing_type][patient_data.patient] = {'y_true': [], 'y_pred': []}
                for i in range(n_experiments):
                    print(f'\nRepetition {i+1} using \'{preprocessing_type}\' trials from patient \'{patient_data.patient}\' ({X.shape[1]} channels, {X.shape[0]} trials)')
                    print(f'Classification with {classifier}...')
                    accuracy, y_true, y_pred, intermediate_y_true, intermediate_y_pred = classification(classifier,
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
                    intermediate_results[classifier][preprocessing_type][patient_data.patient]['y_true'].append(intermediate_y_true)
                    intermediate_results[classifier][preprocessing_type][patient_data.patient]['y_pred'].append(intermediate_y_pred)
                if save_results:
                    directory = f'results/{task}/{classifier}/{preprocessing_type}{classification_type}'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    file = f'{directory}/{patient_data.trials_fileID}_results.pkl'
                    with open(file, 'wb+') as f:
                        pkl.dump(results[classifier][preprocessing_type][patient_data.patient], f)
                    print(f'Results stored in \'{file}\'')
                    if save_intermediate_results:
                        # Store 'intermediate_results', i.e. results after each 5 epochs
                        file = f'{directory}/{patient_data.trials_fileID}_results_learningcurve.pkl'
                        with open(file, 'wb+') as f:
                            pkl.dump(intermediate_results, f)
    return results


def pretrainEEGNet(patient_IDs,
                    labels,
                    task,
                    sampling_rate,
                    trial_window_start,
                    trial_window_stop,
                    model_name,
                    batch_size=5,
                    epochs=25,
                    verbose=0):
    eegnet_kwargs = {
        'n_samples': int((trial_window_stop-trial_window_start) * sampling_rate),
        'dropoutRate': 0.5,
        'kernLength': int(sampling_rate/2),
        'norm_rate': 0.25, 
        'dropoutType': 'Dropout',
        'F1': 8,
        'D': 2,
        'F2': 16
        }
    for pID in patient_IDs:
        patient_data = PatientDataMapper(pID, task)
        # Path where trials are stored
        trials_path = f'data/{task}/{patient_data.trials_fileID}/{patient_data.trials_fileID}_CAR_trials.pkl'
        # Path to store pretrained model
        model_path = f'models/{task}/{patient_data.trials_fileID}/{model_name}'
        # Define mapping from indices to labels (differs per patient)
        labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
        # Load trials for specific patient and task
        X, y = load_trials(trials_path)
        print(f'\nStart pretraining on rest vs. active {task[9:]} data for patient {patient_data.patient}')
        eegnet = EEGNet_tf_Classifier(X, y, labelsdict, n_channels=X.shape[1], **eegnet_kwargs)
        eegnet.pretrain(model_path, batch_size=batch_size, epochs=epochs, verbose=verbose)

def finetuneEEGNet(patient_IDs,
                    labels,
                    task,
                    n_experiments,
                    sampling_rate,
                    trial_window_start,
                    trial_window_stop,
                    model_name,
                    save_results,
                    save_intermediate_results,
                    make_plots):
    # Hyperparameters of EEGNet
    eegnet_kwargs = {
        'n_samples': int((trial_window_stop-trial_window_start) * sampling_rate),
        'dropoutRate': 0.5,
        'kernLength': int(sampling_rate/2),
        'norm_rate': 0.25, 
        'dropoutType': 'Dropout',
        'F1': 8,
        'D': 2,
        'F2': 16
        }
    results = dict()
    intermediate_results = dict()
    for pID in patient_IDs:
        patient_data = PatientDataMapper(pID, task)
        results[patient_data.patient] = {'y_true': [], 'y_pred': []}
        intermediate_results = {'y_true': [], 'y_pred': []}
        # Path where trials are stored
        trials_path = f'data/{task}/{patient_data.trials_fileID}/{patient_data.trials_fileID}_CAR_trials.pkl'
        # Path in which pretrained model is stored
        if task == 'gestures':
            model_path = f'models/pretrain_gestures/{patient_data.trials_fileID}/{model_name}'
        elif task == 'small_gestures':
            model_path = f'models/pretrain_small_gestures/{patient_data.trials_fileID}/{model_name}'
        elif task == 'phonemes' or 'phonemes_noRest':
            model_path = f'models/pretrain_phonemes/{patient_data.trials_fileID}/{model_name}'
        # Define mapping from indices to labels (differs per patient)
        labelsdict = {patient_data.label_indices[i]: labels[i] for i in range(len(labels))}
        # Load trials for specific patient and task
        X, y = load_trials(trials_path)
        for i in range(n_experiments):
            print(f'\nRepetition {i+1} finetuning \'{model_name}\' on \'{task}\' for patient \'{patient_data.patient}\' ({X.shape[1]} channels, {X.shape[0]} trials)')
            eegnet = EEGNet_tf_Classifier(X, y, labelsdict, n_channels=X.shape[1], **eegnet_kwargs)
            acc, y_true, y_pred, intermediate_y_trues, intermediate_y_preds = eegnet.finetune(model_path, make_plots, save_intermediate_results=save_intermediate_results)
            results[patient_data.patient]['y_true'].append(y_true)
            results[patient_data.patient]['y_pred'].append(y_pred)
            intermediate_results['y_true'].append(intermediate_y_trues)
            intermediate_results['y_pred'].append(intermediate_y_preds)

        if save_results:
            directory = f'results/finetuned_{task}/{model_name}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = f'{directory}/{patient_data.trials_fileID}_results.pkl'
            with open(file, 'wb+') as f:
                pkl.dump(results[patient_data.patient], f)
            print(f'Results stored in \'{file}\'')
            if save_intermediate_results:
                # Store 'intermediate_results', i.e. results after each 5 epochs
                file = f'{directory}/{patient_data.trials_fileID}_results_learningcurve.pkl'
                with open(file, 'wb+') as f:
                    pkl.dump(intermediate_results, f)
    return results
            

def pca_visualization(trials_path):
    X, y = load_trials(trials_path)
    print(X.shape)
    
    X_second_dim = 1
    for i in range(1, len(X.shape)):
        X_second_dim *= X.shape[i]
    X = X.reshape(X.shape[0], X_second_dim).astype(np.float32)
    print(X.shape)

    Xt = PCA(n_components=2).fit_transform(X)
    plt.scatter(Xt[:,0], Xt[:,1], c=y)
    plt.show()


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
        print(f'(Accuracy averaged over {len(results)} experiments')
        results = np.array(results)
        return np.mean(results), np.std(results)
    else:
        accs = []
        n_experiments = len(results['y_true'])
        for i in range(n_experiments):
            accs.append(accuracy_score(results['y_true'][i], results['y_pred'][i]))
        accs = np.array(accs)
        return np.mean(accs), np.std(accs)

