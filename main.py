from utils import *

if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Wherer does trial window start relative to VOT (seconds)
    trial_window_start = -0.5
    # Wherer does trial window stop relative to VOT (seconds)
    trial_window_stop = 0.5
    # Either binary (rest vs. active) or multi-class classification
    restvsactive = True
    if restvsactive:
        labels = ['Rest', 'Active']
        classification_type = '_RvA'
    else:
        # Spoken phonemes
        labels = ['/p/', '/oe/', '/a/', '/k/', 'Rest']
        classification_type = ''
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6','7','8']
    # patient_IDs = ['1']
    # Type of preprocessing/features to extract
    # preprocessing_types = ['highgamma', 'allbands', 'broadband40-150', 'articleHFB']
    preprocessing_types = ['delta', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma', 'allbands', 'broadband40-150', 'articleHFB']
    # preprocessing_types = ['delta', 'theta', 'alpha']
    # Define which classifiers to experiment with: 'STMF' / 'SVM' / 'kNN' / 'RF' / 'EEGNet' / 'LSTM'
    classifiers = ['STMF']
    # Number of experiments to average accuracy over 
    # (only useful for non-deterministic classifiers)
    n_experiments = 1
    
    # Which functionalities to execute (True/False)
    preprocess = True
    create_trials = True
    classify = True
    make_plots = False
    save_results = True
    plot_results = False
 
    if preprocess:
        for pID in patient_IDs:
            patient_data = PatientDataMapper(pID)
            for ptype in preprocessing_types:
                ecog_data, valid_points, break_points = preprocessing(patient_data, 
                                                                      sampling_rate,
                                                                      buffer, 
                                                                      ptype)
                if create_trials:
                    trials_path = f'data/{patient_data.patient}/{patient_data.patient}_{ptype}{classification_type}_trials.pkl'
                    trials_creation(patient_data,
                                    ecog_data,
                                    valid_points,
                                    break_points,
                                    sampling_rate,
                                    trials_path,
                                    trial_window_start,
                                    trial_window_stop,
                                    restvsactive)
    
    if classify:
        accuracies = classification_loop(patient_IDs, 
                                         preprocessing_types,
                                         classifiers,
                                         classification_type,
                                         n_experiments,
                                         sampling_rate,
                                         trial_window_start,
                                         trial_window_stop,
                                         make_plots,
                                         save_results,
                                         LOO=True)
        print_results(accuracies, n_experiments)

    if plot_results:
        # plot_features_results(classifiers, preprocessing_types, patient_IDs)
        plot_classifier_results(patient_IDs)

    # TODO:
    # - Try LSTM without dropout
    # - Test multiple k's for k-NN
    # - Check if batch size matters for EEGNet(?)
    # - Figure out how to deal with interpretation of results: rest class might be 100% accurate and influence total accuracy
    #       Maybe use F1 score, or visualize confusion matrix, but that's not possible to do for all experiments
    # - Find better way to store results
    #       Might be better to put in classifier (base?) class
    #       'Add results' functionality needed with how results are currently stored
    #       >>> Might be better to store y_pred and y_true, to be able to reconstruct CM from stored results
    # - Interpretation of kernel weights EEGNet
    # - Start implementation of EEGNet to pretrain on active vs. rest
    
    # - In patient '4' trials data: 5 trials have -1 at 3rd column that don't have -1 at 2nd column
    #   while having normal label and normal VOT. Are those bad trials or not? (Not the case for other patients)

    # - Substitute last layer(s) of EEGNet by an LSTM
    # - Test LSTM alone (without feature extraction layers)

    # Probably better to create 1 script for classifcation with DL models
    # And for each neural network a separate class just for init and forward

    # Questions:
    # - Should I try different windowsizes?
    # - Should I implement simple FNN? (for preprocessed data)
    # - For hyperparameter optimization: just do different LOO runs? (for example for different k's in kNN)
    # - Is the class imbalance in RvA a problem?