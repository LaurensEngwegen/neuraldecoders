from utils import *

import os
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from scipy import stats
import pickle as pkl
import matplotlib.pyplot as plt

# Increase font size in plots
plt.rcParams.update({'font.size': 16})

# Functions to get info

def get_bar_positions(n_params, width):
    # Make 'pos' a list of len=n_params with values around 0
    pos = []
    if n_params % 2 == 0: # even number of params
        begin = round(-0.5*width-((n_params-2)/2)*width, 2)
        end = round(0.5*width+((n_params-1)/2)*width, 2)
        for i in np.arange(begin, 0, width):
            pos.append(round(i,2))
        for i in np.arange(0.5*width, end, width):
            pos.append(round(i,2))
    else: # odd number of params
        begin = round(0-((n_params-1)/2)*width, 2)
        end = round(((n_params+1)/2)*width, 2)
        for i in np.arange(begin, 0, width):
            pos.append(round(i,2))
        for i in np.arange(0, end, width):
            pos.append(round(i,2))
    return pos

def get_classifiers_info(task, classifiers=None, pretrain_model=None):
    result_info = {
        'STMF': {'classifier': 'STMF', 'ptype': 'gamma'}
    }
    if task == 'phonemes' or task == 'phonemes_noRest':
        if task == 'phonemes':
            n_classes = '5'
        else:
            n_classes = '4'
        title = f'Accuracy of different classifiers on {n_classes}-class phoneme classification'
        patient_IDs = ['1','2','3','4','5','6','7','8']
        result_info['kNN'] = {'ptype': 'gamma', 'classifier': 'kNN11'}
        result_info['SVM'] = {'ptype': 'gamma', 'classifier': 'SVM'}
        result_info['MLP (gamma)'] = {'ptype': 'gamma', 'classifier': 'FFN128-64'}
        result_info['MLP (CAR)'] =  {'classifier': 'FFN128-64', 'ptype': 'CAR'}
    else: # gestures or small_gestures
        title = f'Accuracy of different classifiers on 4-class gesture classification'
        patient_IDs = ['9','10','11','12','13']
        result_info['kNN'] = {'ptype': 'gamma', 'classifier': 'kNN3'}
        result_info['SVM'] = {'ptype': 'gamma', 'classifier': 'SVM'}
        result_info['MLP (gamma)'] = {'ptype': 'gamma', 'classifier': 'FFN64-32'}
        result_info['MLP (CAR)'] =  {'classifier': 'FFN128-64', 'ptype': 'CAR'}
    
    result_info['EEGNet'] = {'ptype': 'CAR', 'classifier': 'EEGNet'}
    if pretrain_model is not None:
        result_info['EEGNet finetuned'] = {'ptype': 'CAR', 'classifier': 'EEGNet finetuned'}
    
    if classifiers is not None:
        returndict = dict()
        for classifier in classifiers:
            returndict[classifier] = result_info[classifier]
    else:
        returndict = result_info

    return returndict, patient_IDs, title

def write_all_accuracies(pretrain_model='second_model'):
    acc_directory = 'results/accuracies'
    for task in ['phonemes_noRest', 'small_gestures']:
        save_path = f'{acc_directory}/{task}/accuracies.txt'
        classifier_info, patient_IDs, _ = get_classifiers_info(task, pretrain_model)
        # Get patient names
        patient_labels = []
        for patient_ID in patient_IDs:
            patient_labels.append(PatientDataMapper(patient_ID, task).patient)
        # Read results
        # For all classifiers defined in classifier_info
        for key in classifier_info.keys():
            accs = []
            classifier = classifier_info[key]['classifier']
            preprocessing_type = classifier_info[key]['ptype']
            for patient in patient_labels:
                # Define location where results are stored
                if classifier == 'EEGNet finetuned':
                    results_directory = f'results/finetuned_{task}/{pretrain_model}'
                else:
                    results_directory = f'results/{task}/{classifier}/{preprocessing_type}'
                results_file = f'{results_directory}/{patient}_results.pkl'
                acc, _ = get_accuracy(results_file)
                with open(save_path, 'a+') as f:
                    f.write(f'{key}\t{patient}\t{acc}\n')

def do_paired_ttest(pretrain_model='model_100epochs'):
    acc_directory = 'results/accuracies'
    for task in ['phonemes_noRest', 'small_gestures']:
        print(f'\nTask: {task}')
        acc_path = f'{acc_directory}/{task}/accuracies.txt'
        with open(acc_path) as f:
            reader = csv.reader(f, delimiter='\t')
            acc_list = list(reader)
        classifier_info, patient_IDs, _ = get_classifiers_info(task, pretrain_model)
        classifiers = list(classifier_info.keys())
        patients = []
        for patient_ID in patient_IDs:
            patients.append(PatientDataMapper(patient_ID, task).patient)
        # Create dict: {'classifier': [accuracies]}
        acc_dict = dict()
        for item in acc_list:
            if item[0] not in acc_dict.keys():
                acc_dict[item[0]] = []
            acc_dict[item[0]].append(float(item[2]))

        for i in range(len(classifiers)):
            # print(f'Shapiro test for {classifiers[i]}: {stats.shapiro(acc_dict[classifiers[i]])}')
            for j in range(i+1, len(classifiers)):
                testresult = stats.ttest_rel(acc_dict[classifiers[i]], acc_dict[classifiers[j]])
                print(f'{classifiers[i]} - {classifiers[j]}: {testresult.pvalue}')

    


# Functions to create barplots

def plot_features_results(classifier, preprocessing_types, patient_IDs, restvsactive=False, task='phonemes', save_fig=False, anonymized=True):
    if restvsactive:
        preprocessing_types = [ptype+'_RvA' for ptype in preprocessing_types]
    # Define plot variables
    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID, task).patient)
    x_axis = np.arange(len(patient_IDs))
    # Put +1 here to be able to include article accuracy
    n_params = len(preprocessing_types)
    width = 0.1
    pos = get_bar_positions(n_params, width)
    
    # Read all result data
    all_data = []
    for preprocessing_type in preprocessing_types:
        data = []
        directory = f'results/{task}/{classifier}/{preprocessing_type}'
        for patient_ID in patient_IDs:
            patient_data = PatientDataMapper(patient_ID, task)
            file = f'{directory}/{patient_data.trials_fileID}_results.pkl'
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
    # plot_article_acc(x_axis, pos[len(pos)-1]+0.05, width, restvsactive, task) # To include hardcoded article results
    
    save_path = f'figures/{task}/{classifier}_differentfeatures'
    # save_path = f'figures/{task}/{classifier}_differentfeatures_wArticle'
    if task == 'phonemes':
        if restvsactive:
            title = f'Accuracy of {classifier} with different features on rest vs. active phoneme'
            save_path = f'figures/{task}/{classifier}_RvA_differentfeatures'
        else:
            title = f'Accuracy of {classifier} with different features on 5-class phonemes'
    elif task == 'phonemes_noRest':
        title = f'Accuracy of {classifier} with different features on 4-class phonemes'
    else: # gestures or small_gestures
        title = f'Accuracy of {classifier} with different features on 4-class gestures'
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Patient')
    if anonymized:
        plt.xticks(x_axis, patient_IDs)
        save_path = f'{save_path}_anonym'
    else:
        plt.xticks(x_axis, patient_labels)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}.png', format='png')
        print(f'Figure saved at \'{save_path}\'')
    plt.show()
    
def plot_clf_optimization(classifier, preprocessing_type, patient_IDs, restvsactive=False, task='phonemes', save_fig=False, anonymized=True):
    # k's tested for kNN
    if classifier == 'kNN':
        params = ['3','5','7','9','11','13','15','17','19']
        labelprefix = '$k$ ='
        classifier_title = '$k$NN'
    # Architectures tested for FFN
    elif classifier == 'FFN':
        params = ['256-128','128-64','64-32','32-16']
        labelprefix = 'MLP'
        classifier_title = 'MLP'
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
    pos = get_bar_positions(n_params, width)
    
    # Read all result data
    all_data = []
    for tested_clf in tested_classifiers:
        data = []
        directory = f'results/{task}/{tested_clf}/{preprocessing_type}'
        for patient_ID in patient_IDs:
            patient_data = PatientDataMapper(patient_ID, task)
            file = f'{directory}/{patient_data.trials_fileID}_results.pkl'
            acc, _ = get_accuracy(file)
            data.append(acc)
        all_data.append(data) # Patients x Preprocessing_types
    # Barplot
    plt.figure(figsize=(12,8))
    for i, data in enumerate(all_data):
        plt.bar(x_axis + pos[i], data, width=width, label=f'{labelprefix} {params[i]}')
    if task == 'phonemes':
        if restvsactive:
            title = f'Fundamental optimization  of {classifier_title} ({p_type}) on active vs. rest'
        else:
            title = f'Fundamental optimization  of {classifier_title} ({p_type}) on 5-class phonemes'
    elif task == 'phonemes_noRest':
        title = f'Fundamental optimization of {classifier_title} ({p_type}) on 4-class phonemes'
    else: # gestures or small_gestures
        title = f'Fundamental optimization  of {classifier_title} ({p_type}) on 4-class gestures'
    save_path = f'figures/{task}/{classifier}_{p_type}_optimization'
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Patient')
    if anonymized:
        plt.xticks(x_axis, patient_IDs)
        save_path = f'{save_path}_anonym'
    else:
        plt.xticks(x_axis, patient_labels)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}.png', format='png')
        print(f'Figure saved at \'{save_path}\'')
    plt.show()

def plot_classifier_results(task='phonemes', pretrain_model=None, save_fig=False, anonymized=True):
    classifier_info, patient_IDs, title = get_classifiers_info(task, pretrain_model)
    
    patient_labels = []
    for patient_ID in patient_IDs:
        patient_labels.append(PatientDataMapper(patient_ID, task).patient)
    x_axis = np.arange(len(patient_IDs))
    width = 0.1
    # Add +1 to include article acc in plot
    n_params = len(classifier_info)
    pos = get_bar_positions(n_params, width)

    all_accs, all_stds, titles = [], [], []
    for key in classifier_info.keys():
        accs, stds = [], []
        classifier = classifier_info[key]['classifier']
        preprocessing_type = classifier_info[key]['ptype']
        if classifier == 'EEGNet finetuned':
            directory = f'results/finetuned_{task}/{pretrain_model}'
        else:
            directory = f'results/{task}/{classifier}/{preprocessing_type}'
        for patient in patient_labels:
            file = f'{directory}/{patient}_results.pkl'
            acc, std = get_accuracy(file)
            accs.append(acc)
            stds.append(std)
        all_accs.append(accs)
        all_stds.append(stds)
        titles.append(key)

    save_path = f'figures/{task}/all_clfs_barplot'
    plt.figure(figsize=(12,8))
    # Add +1 to n_params to include article acc.
    # plot_article_acc(x_axis, pos[len(pos)-1], width, task=task)
    for i, accuracies in enumerate(all_accs):
        plt.bar(x_axis+pos[i], accuracies, yerr=all_stds[i], width=width, label=titles[i])
    plt.title(title)
    if anonymized:
        plt.xticks(x_axis, patient_IDs)
        save_path = f'{save_path}_anonym'
    else:
        plt.xticks(x_axis, patient_labels)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    if save_fig:
        
        plt.savefig(f'{save_path}.png', format='png')
        print(f'Figure saved at \'{save_path}\'')
    plt.show()

def plot_article_acc(x_axis, pos, width, restvsactive=False, task='phonemes', anonymized=True):
    if task == 'phonemes_noRest':
        article_accuracies = [0.814, 0.621, 0.77, 0.63, 0.761, 0, 0, 0]
        if restvsactive:
            article_accuracies = [0.831, 1, 1, 0.9, 0.667, 0, 0, 0]
        label = 'Article STMF'
    else:
        return
    plt.bar(x_axis+pos, article_accuracies, width=width, label=label, color='lightgrey')


# Functions to create accuracy during training of neural nets and second plot for classifier comparison

def plot_learning_curve(task, model='EEGNet', pretrain_model='model_25epochs', save_fig=False, anonymized=True):
    info, _, _ = get_classifiers_info(task)
    classifier = info[model]['classifier']
    p_type = info[model]['ptype']
    if task =='phonemes_noRest' or task == 'finetuned_phonemes_noRest':
        patient_IDs = ['1','2','3','4','5','6','7','8']
        task_title = '4-phonemes'
    elif task == 'gestures' or task == 'finetuned_gestures' or task == 'small_gestures' or task == 'finetuned_small_gestures':
        patient_IDs = ['9','10','11','12','13']
        task_title = '4-gestures'

    directory = f'results/{task}/{classifier}/{p_type}'
    if task[:9] == 'finetuned': # finetuned_phonemes_noRest or finetuned_small_gestures
        directory = f'results/{task}/{pretrain_model}'
        model_title = f'finetuned {classifier}'
        save_path = f'figures/{task}/EEGNet_finetuned_{pretrain_model}_learningcurve'
    else:
        model_title = model
        save_path = f'figures/{task}/{classifier}_{p_type}_learningcurve'

    plt.figure(figsize=(12,8))

    for patient_ID in patient_IDs:
        patient_info = PatientDataMapper(patient_ID, task)
        file_path = f'{directory}/{patient_info.patient}_results_learningcurve.pkl'
        with open(file_path, 'rb') as f:
            results = pkl.load(f)
        # y_true and y_pred contain a list of 5 independent runs with true/predicted y's in dictionary for different epochs (0/5/.../25)
        if task[:9] == 'finetuned':
            y_true = results['y_true']
            y_pred = results['y_pred']
        else:
            y_true = results[classifier][p_type][patient_info.patient]['y_true']
            y_pred = results[classifier][p_type][patient_info.patient]['y_pred']
        mean_per_epoch, std_per_epoch, epochs = [], [], []
        # print(f'\n{patient_info.patient}')
        epoch_counter = 0
        for key in y_true[0]:
            epochs.append(epoch_counter)
            accuracy = []
            for i in range(len(y_true)):
                accuracy.append(accuracy_score(y_true[i][key], y_pred[i][key]))
            # print(f'{key}: {np.mean(accuracy)}')
            mean_per_epoch.append(np.mean(accuracy))
            std_per_epoch.append(np.std(accuracy))
            epoch_counter += 5
        mean_per_epoch = np.array(mean_per_epoch)
        std_per_epoch = np.array(std_per_epoch)
        if anonymized:
            label = patient_info.ID
        else:
            label = patient_info.patient
        plt.plot(epochs, mean_per_epoch, label=label)
        plt.fill_between(epochs, mean_per_epoch-std_per_epoch, mean_per_epoch+std_per_epoch, alpha=0.1)

    if anonymized:
        save_path = f'{save_path}_anonym'
    plt.title(f'Learning curve of {model_title} on {task_title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    end = (len(mean_per_epoch)*5)-5 # Starts at 0, steps of 5
    plt.xticks(np.arange(0,end+1,10))
    plt.xlim(0,end)
    plt.yticks(np.arange(0,1.01,0.1))
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}.png', format='png')
        print(f'Figure saved at \'{save_path}\'')
    plt.show()

def new_clf_plot(task='phonemes_noRest', classifiers=None, pretrain_model=None, barplot=False, save_fig=False, anonymized=True):
    classifier_info, patient_IDs, title = get_classifiers_info(task, classifiers=classifiers, pretrain_model=pretrain_model)
    n_params = len(patient_IDs)
    x_axis = np.arange(len(classifier_info))
    width = 0.08
    pos = get_bar_positions(n_params, width)
    save_path = f'figures/{task}/all_clfs_lineplot'

    plt.figure(figsize=(11,8))
    for i, patient_ID in enumerate(patient_IDs):
        patient_info = PatientDataMapper(patient_ID, task)
        accs, stds = [], []
        # For all classifiers defined in classifer_info
        for key in classifier_info.keys():
            classifier = classifier_info[key]['classifier']
            preprocessing_type = classifier_info[key]['ptype']
            # Define location where results are stored
            if classifier == 'EEGNet finetuned':
                directory = f'results/finetuned_{task}/{pretrain_model}'
            else:
                directory = f'results/{task}/{classifier}/{preprocessing_type}'
            file = f'{directory}/{patient_info.patient}_results.pkl'
            acc, std = get_accuracy(file)
            accs.append(acc)
            stds.append(std)
        if anonymized:
            label = patient_info.ID
        else:
            label = patient_info.patient        
        if barplot:
            plt.bar(x_axis+pos[i], accs, label=label, width=width, alpha=0.8)
        else:
            plt.plot(np.arange(len(accs)), accs, label=label, marker='p', markersize=9, linestyle=(0, (1, 5)), linewidth=1.2, alpha=0.7)
    
    if barplot:
        plt.plot([-0.5, 6.5], [0.25, 0.25], linestyle=':', c='black', alpha=0.5)
        save_path = f'{save_path}_bar'
    else:
        plt.plot(np.arange(len(accs)), [0.25]*len(accs), linestyle='dashed', c='black', alpha=0.5)
        # plt.plot([-0.5,1.5], [0.25,0.25], linestyle='dashed', c='black', alpha=0.5)
    
    x_axis = np.arange(len(classifier_info))
    if anonymized:
        save_path = f'{save_path}_anonym'
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(x_axis, classifier_info.keys())
    plt.yticks(np.arange(0,1.01,0.1))
    plt.ylim(bottom=0.0)
    plt.legend(loc = 6, bbox_to_anchor = (1, 0.5))
    plt.grid(alpha=0.35)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{save_path}.png', format='png')
        print(f'Figure saved at \'{save_path}\'')
    plt.show()


