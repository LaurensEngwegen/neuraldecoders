from utils import *
from plots import *

def create_plots():
    to_plot = 'allclassifiers'

    # Need to define which patients and tasks to plot results from
    IDs = ['1','2','3','4','5','6','7','8']
    task = 'phonemes_noRest'
    # IDs = ['9','10','11','12','13']
    # task = 'small_gestures'

    if to_plot == 'features':

        classifier = 'STMF'
        p_type = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'allbands']
        plot_features_results(classifier, p_type, IDs, task=task, restvsactive=False, save_fig=True, anonymized=True)

    elif to_plot == 'optimization':
        classifier = 'kNN'
        p_type = 'gamma'
        plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)
        classifier = 'FFN'
        p_type = 'gamma'
        plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)
        classifier = 'FFN'
        p_type = 'CAR'
        plot_clf_optimization(classifier, p_type, IDs, task=task, save_fig=True, anonymized=True)

    elif to_plot == 'allclassifiers':
        pretrain_model = 'second_model'

        # Barplot with patients on x-axis
        # tasks = ['small_gestures', 'phonemes_noRest']
        # for task in tasks:
        #     plot_classifier_results(task, pretrain_model=None, save_fig=False, anonymized=True)
        
        # Lineplot or barplot with classifiers on x-axis (barplot not recommended)
        tasks = ['small_gestures', 'phonemes_noRest']
        for task in tasks:
            # classifiers = ['EEGNet', 'EEGNet finetuned']
            classifiers = None # For all decoding strategies
            new_clf_plot(task, classifiers, pretrain_model=None, barplot=False, save_fig=False, anonymized=True)

    elif to_plot == 'learningcurve':
        pretrain_model = 'second_model'
        tasks = ['phonemes_noRest', 'small_gestures']
        for task in tasks:
            model = 'MLP (gamma)'
            plot_learning_curve(task, model, pretrain_model=pretrain_model, save_fig=True, anonymized=True)


if __name__ == '__main__':
    # Sampling rate of signal
    sampling_rate = 512
    # Number of samples at beginning and end of file that should be excluded
    buffer = 64
    # Which task ('phonemes' (5-class), 'phonemes_noRest', 'gestures', 'small_gestures')
    # For pretraining: 'pretrain_phonemes' or 'pretrain_small_gestures'
    task = 'phonemes_noRest'
    # Patient data to use
    patient_IDs = ['1','2','3','4','5','6','7','8'] # Phonemes
    # patient_IDs = ['10']
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
    save_results = False
    plot_results = False # The ones defined in create_plots()

    # Either binary (rest vs. active) or multi-class classification
    restvsactive = False

    # Transfer learning
    pretrain = False # Set batch_size and epochs in function call below
    finetune = False
    model_name = 'second_model'

    # Second model = Minimal number of rest vs. active trials + batch_size=5 + epochs=100
    # model_100epochs = The same but trained on enlarged dataset

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

    if plot_results:
        create_plots()

    if pca:
        for pID in patient_IDs:
            patient_data = PatientDataMapper(pID, task)
            for ptype in preprocessing_types:
                trials_path = f'data/{task}/{patient_data.patient}/{patient_data.patient}_{ptype}{classification_type}_trials.pkl'
                pca_visualization(trials_path)

    # write_all_accuracies(pretrain_model='second_model', to_file=True)
    # do_paired_ttest(pretrain_model='second_model')
    
