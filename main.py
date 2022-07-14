from utils import *
from plots import *

def create_plots():
    pretrain_model = 'model_100epochs' # define for new_clf_plot to include finetuned EEGNet

    IDs = ['1','2','3','4','5','6','7','8']
    task = 'phonemes_noRest'
    # IDs = ['9','10','11','12','13']
    # task = 'small_gestures'

    # classifier = 'STMF'
    # p_type = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'allbands']
    # plot_features_results(classifier, p_type, IDs, task=task, restvsactive=False, save_fig=True, anonymized=True)

    # classifier = 'kNN'
    # p_type = 'gamma'
    # plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)
    # classifier = 'FFN'
    # p_type = 'gamma'
    # plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)
    # classifier = 'FFN'
    # p_type = 'CAR'
    # plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)

    # plot_classifier_results(task, pretrain_model=pretrain_model, save_fig=False, anonymized=True)

    tasks = ['finetuned_phonemes_noRest', 'finetuned_small_gestures']
    for task in tasks:
        model = 'EEGNet'
        # plot_learning_curve(task, model, pretrain_model=pretrain_model, save_fig=True, anonymized=True)

    # write_all_accuracies()
    # do_paired_ttest()

    task = 'phonemes_noRest'
    # classifiers = ['EEGNet', 'EEGNet finetuned']
    classifiers = None
    # new_clf_plot(task, classifiers, pretrain_model=None, barplot=False, save_fig=True, anonymized=True)


if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Which task ('phonemes' (5-class), 'phonemes_noRest', 'gestures', s'mall_gestures')
    # For pretraining: 'pretrain_phonemes' or 'pretrain_gestures'
    task = 'phonemes_noRest'
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6','7','8'] # Phonemes
    # patient_IDs = ['9','10','11','12','13'] # Gestures
    # Type of preprocessing/features to extract
    # preprocessing_types = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'allbands']
    preprocessing_types = ['CAR']
    # Define which classifiers to experiment with: 'STMF' / 'SVM' / 'kNN' / 'FFN' / 'EEGNet' (/ 'LSTM')
    # kNN and FFN need extra information:
    # - kNN: value for k in (1,3,...,19), e.g. ['kNN11']
    # - FFN: number of nodes in hidden layers from (256-128, 128-64, 64-32, 32-16)
    classifiers = ['EEGNet']
    #  Number of experiments to average accuracy over 
    # (only useful for non-deterministic classifiers)
    n_experiments = 5

    # Which functionalities to execute (True/False)
    preprocess = False
    create_trials = False
    classify = False
    make_plots = False # Plot mean freq band and confusion matrix
    save_results = True
    plot_results = False
    # Either binary (rest vs. active) or multi-class classification
    restvsactive = False

    # Transfer learning
    pretrain = False # Set batch_size and epochs in function call below
    finetune = True
    model_name = 'first_model'

    # First model = Minimal number of rest vs. active trials + batch_size=5 + epochs=25
    # Second model = Extended number of rest vs. active trials + batch_size=5 + epochs=25
    # Third model = Extended number of rest vs. active trials + batch_size=32 + epochs=25
    # Fourth model = thirdmodel with epochs=8 / .....

    pca = False

    task_info = {
        'phonemes': {'labels': ['/p/', '/oe/', '/a/', '/k/', 'Rest'], 'windowstart': -0.5, 'windowstop': 0.5},
        'phonemes_noRest': {'labels': ['/p/', '/oe/', '/a/', '/k/'], 'windowstart': -0.5, 'windowstop': 0.5},
        'pretrain_phonemes': {'labels': ['Rest', 'Active'], 'windowstart': -0.5, 'windowstop': 0.5},
        'small_gestures': {'labels': ['G1', 'G2', 'G3', 'G4'], 'windowstart': -0.5, 'windowstop': 0.5},
        'pretrain_small_gestures': {'labels': ['Rest', 'Active'], 'windowstart': -0.5, 'windowstop': 0.5},
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
        print(f'\nClassification of \'{task}\'...\n')
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
                                         save_intermediate_results=True,
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
                    batch_size=5,
                    epochs=100,
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
                                save_intermediate_results=True,
                                make_plots=make_plots)
        # print_results(results, n_experiments) error cause not [classifier][ptype][patient] but just [patient]

    if plot_results:
        create_plots()

    if pca:
        for pID in patient_IDs:
            patient_data = PatientDataMapper(pID, task)
            for ptype in preprocessing_types:
                trials_path = f'data/{task}/{patient_data.patient}/{patient_data.patient}_{ptype}{classification_type}_trials.pkl'
                pca_visualization(trials_path)

    # small_gestures currently running:
        # all features: STMF, SVM
        # CAR: all FFNs, EEGNet

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