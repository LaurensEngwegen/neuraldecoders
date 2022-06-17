from cProfile import label
from site import makepath
from utils import *

if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Which task (phonemes, phonemes_noRest, or gestures)
    # For finetuning: need to set task to either phonemes_noRest or gestures!
    # task = 'phonemes_noRest'
    task = 'phonemes_noRest'
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6','7','8']
    # patient_IDs = ['9','10','11','12','13']
    # patient_IDs = ['2','3']
    # patient_IDs = ['7.2']
    # Type of preprocessing/features to extract
    # preprocessing_types = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'allbands']
    # preprocessing_types = ['gestures_gamma', 'gamma']
    preprocessing_types = ['CAR']
    # Define which classifiers to experiment with: 'STMF' / 'SVM' / 'kNN' / ('RF') / 'FFN' / 'EEGNet' / 'LSTM'
    # classifiers = ['FFN256-128','FFN128-64','FFN64-32','FFN32-16']
    # classifiers = ['FFN32-16']
    # classifiers = ['kNN3','kNN5','kNN7','kNN9','kNN11','kNN13','kNN15','kNN17','kNN19', 'STMF', 'SVM']
    # classifiers = ['LSTM32', 'LSTM64', 'LSTM128', 'LSTM256']
    classifiers = ['EEGNet']
    # classifiers = ['STMF', 'SVM']
    #  Number of experiments to average accuracy over 
    # (only useful for non-deterministic classifiers)
    n_experiments = 5
    
    # Which functionalities to execute (True/False)
    preprocess = False
    create_trials = False
    classify = False
    make_plots = False # Mean freq band and confusion matrix
    save_results = True
    plot_results = False
    # Either binary (rest vs. active) or multi-class classification
    restvsactive = False

    # Transfer learning
    pretrain = False
    finetune = True
    model_name = 'third_model'

    pca = False

    task_info = {
        'phonemes': {'labels': ['/p/', '/oe/', '/a/', '/k/', 'Rest'], 'windowstart': -0.5, 'windowstop': 0.5},
        'phonemes_noRest': {'labels': ['/p/', '/oe/', '/a/', '/k/'], 'windowstart': -0.5, 'windowstop': 0.5},
        'pretrain_phonemes': {'labels': ['Rest', 'Active'], 'windowstart': -0.5, 'windowstop': 0.5},
        'gestures': {'labels': ['G1', 'G2', 'G3', 'G4'], 'windowstart': -1.0, 'windowstop': 2.6},
        'pretrain_gestures': {'labels': ['Rest', 'Active'], 'windowstart': -1.0, 'windowstop': 2.6}
    }

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
                    # Set trials_path to None to not save created trials
                    trials_path = f'data/{task}/{patient_data.trials_fileID}/{patient_data.trials_fileID}_{ptype}{classification_type}_trials.pkl'
                    trials_creation(patient_data,
                                    ecog_data,
                                    valid_points,
                                    break_points,
                                    sampling_rate,
                                    trials_path=trials_path,
                                    trial_window_start=task_info[task]['windowstart'],
                                    trial_window_stop=task_info[task]['windowstop'],
                                    restvsactive=restvsactive,
                                    task=task)
    
    if classify:
        results = classification_loop(patient_IDs, 
                                         preprocessing_types,
                                         classifiers,
                                         classification_type,
                                         task_info[task]['labels'],
                                         n_experiments,
                                         sampling_rate,
                                         task_info[task]['windowstart'],
                                         task_info[task]['windowstop'],
                                         make_plots,
                                         save_results,
                                         LOO=True,
                                         task=task)
        print_results(results, n_experiments)

    if pretrain:
        pretrainEEGNet(patient_IDs=patient_IDs,
                    labels=task_info[task]['labels'],
                    task=task,
                    sampling_rate=sampling_rate,
                    trial_window_start=task_info[task]['windowstart'],
                    trial_window_stop=task_info[task]['windowstop'],
                    model_name=model_name,
                    batch_size=32,
                    verbose=1)
    
    if finetune:
        results = finetuneEEGNet(patient_IDs=patient_IDs,
                                labels=task_info[task]['labels'],
                                task=task,
                                n_experiments=n_experiments,
                                sampling_rate=sampling_rate,
                                trial_window_start=task_info[task]['windowstart'],
                                trial_window_stop=task_info[task]['windowstop'],
                                model_name=model_name,
                                save_results=save_results,
                                make_plots=make_plots)
        # print_results(results, n_experiments) error cause not [classifier][ptype][patient] but just [patient]

    if plot_results:
        # plot_features_results(classifiers, preprocessing_types, patient_IDs, restvsactive, task=task)
        # plot_clf_optimization(['kNN', 'FFN'], 'gamma', patient_IDs, restvsactive, task=task)
        plot_classifier_results(patient_IDs, task=task)

    if pca:
        for pID in patient_IDs:
            patient_data = PatientDataMapper(pID, task)
            for ptype in preprocessing_types:
                trials_path = f'data/{task}/{patient_data.patient}/{patient_data.patient}_{ptype}{classification_type}_trials.pkl'
                pca_visualization(trials_path)

    # TODO:
    # V Add print info (nr. of experiments) when creating plots
    # - t-SNE for data visualization
    # - Fix HFB plotting (to be compatible with different number of classes)
    # - Run all neural networks 5 more times (to get 10 in total)
    # - Interpretation of kernel weights EEGNet
    # - Change plot titles (on basis of task)
    # - Run EEGNet with larger batchsize? Probably doesn't matter/not going to work because of the small number of trials

    # - Multiple classes only available for 3 patients, including 7&8: the patients with noisy data

    # - Drop too early and too late trials???
    
    # - In patient '4' trials data: 5 trials have -1 at 3rd column that don't have -1 at 2nd column
    #   while having normal label and normal VOT. Are those bad trials or not? (Not the case for other patients)

    # Probably better to create 1 script for classifcation with DL models
    # And for each neural network a separate class just for init and forward

    # Questions:
    # - Should I not use/look at F1 score (instead of accuracy)?

    # Correct results (without removing 'too early' and 'too late' trials):
    # - Phonemes: kNN, STMF, SVM, EEGNet, FFNgamma bash2, FFNCAR bash3
    # - Phonemes_noRest: kNN, STMF, SVM, EEGNet, FFNgamma, FFNCAR
    # - Phonemes RvA: STMF
    # - Gestures: kNN, STMF, SVM, (EEGNet: results with old code, but too early/late removed, so should be correct), FFNgamma, FFNCAR


    # Should I still use categorical crossentropy loss when pretraining? Instead of binary crossentropy