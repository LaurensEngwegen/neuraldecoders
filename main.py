from utils import *

if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Which task (phonemes or gestures)
    task = 'phonemes'
    # task = 'gestures'
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6','7','8']
    # patient_IDs = ['1','2','3','5','6','7','8']
    # patient_IDs = ['9','10','11','12','13']
    # patient_IDs = ['4']
    # Type of preprocessing/features to extract
    # preprocessing_types = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'allbands']
    preprocessing_types = ['CAR']
    # Define which classifiers to experiment with: 'STMF' / 'SVM' / 'kNN' / ('RF') / 'EEGNet' / 'LSTM'
    classifiers = ['FFN256-128','FFN128-64','FFN64-32','FFN32-16']
    # classifiers = ['kNN3','kNN5','kNN7','kNN9','kNN11','kNN13','kNN15','kNN17','kNN19']
    # classifiers = ['LSTM32', 'LSTM64', 'LSTM128', 'LSTM256']
    # classifiers = ['EEGNet']
    # classifiers = ['STMF', 'SVM']
    #  Number of experiments to average accuracy over 
    # (only useful for non-deterministic classifiers)
    n_experiments = 10
    
    # Which functionalities to execute (True/False)
    preprocess = False
    create_trials = False
    classify = True
    make_plots = False
    save_results = True
    plot_results = False
    # Either binary (rest vs. active) or multi-class classification
    restvsactive = False
    
    if task == 'phonemes':
        labels = ['/p/', '/oe/', '/a/', '/k/', 'Rest']
        # Where does trial window start relative to VOT (seconds)
        trial_window_start = -0.5
        # Where does trial window stop relative to VOT (seconds)
        trial_window_stop = 0.5
    else: # gestures
        labels = ['G1', 'G2', 'G3', 'G4'] # No rest class
        # Where does trial window start relative to MOT (seconds)
        trial_window_start = -1.0
        # Where does trial window stop relative to MOT (seconds)
        trial_window_stop = 2.6
    
    if restvsactive:
        labels = ['Rest', 'Active']
        classification_type = '_RvA'
    else:
        classification_type = ''
 
    if preprocess:
        for pID in patient_IDs:
            patient_data = PatientDataMapper(pID, task)
            for ptype in preprocessing_types:
                ecog_data, valid_points, break_points = preprocessing(patient_data, 
                                                                      sampling_rate,
                                                                      buffer, 
                                                                      ptype,
                                                                      task)
                if create_trials:
                    trials_path = f'data/{task}/{patient_data.patient}/{patient_data.patient}_{ptype}{classification_type}_trials.pkl'
                    trials_creation(patient_data,
                                    ecog_data,
                                    valid_points,
                                    break_points,
                                    sampling_rate,
                                    trials_path=trials_path,
                                    trial_window_start=trial_window_start,
                                    trial_window_stop=trial_window_stop,
                                    restvsactive=restvsactive,
                                    task=task)
    
    if classify:
        results = classification_loop(patient_IDs, 
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
                                         task=task)
        print_results(results, n_experiments)

    if plot_results:
        # plot_features_results(classifiers, preprocessing_types, patient_IDs, restvsactive, task='phonemes')
        # plot_clf_optimization(['kNN', 'FFN'], 'gamma', patient_IDs, restvsactive, task='gestures')
        plot_classifier_results(patient_IDs, task='gestures')

    # TODO:
    # - Run 4-class phoneme classification
    # - Fix HFB plotting (to be compatible with different number of classes)
    # - FFN: run 5 more times for 1,2,3,5,6,7,8
    # - Run FFN also for CAR data (to compare with EEGNet & LSTM)
    # - Interpretation of kernel weights EEGNet
    
    # - Drop too early and too late trials???
    
    # - In patient '4' trials data: 5 trials have -1 at 3rd column that don't have -1 at 2nd column
    #   while having normal label and normal VOT. Are those bad trials or not? (Not the case for other patients)

    # - Substitute last layer(s) of EEGNet by an LSTM

    # Probably better to create 1 script for classifcation with DL models
    # And for each neural network a separate class just for init and forward

    # Questions:
    # - Should I try different windowsizes?
    # - Should I not use/look at F1 score (instead of accuracy)?