import os
import numpy as np
import h5py
import pickle as pkl

class Trials_Creator():
    def __init__(self, 
                 task_path, 
                 ecog_data, 
                 valid_spectra_points,
                 break_points,
                 patient_data,
                 sampling_rate,
                 save_path=None, 
                 time_window_start=-0.5, 
                 time_window_stop=0.5,
                 restvsactive=False):
        # Hyperparameters for trial construction
        self.sampling_rate = sampling_rate
        self.time_window_start = time_window_start # seconds
        self.time_window_stop = time_window_stop # seconds
        self.offset = patient_data.VOT_offset # samples

        # Whether binary (rest vs. active) or multi-class classification
        self.restvsactive = restvsactive

        # Last one should always be rest
        self.label_indices = patient_data.label_indices
        
        # Read task data
        self.task_data = self.read_taskdata(task_path)
        # Downsampling if needed
        if patient_data.sampling_rate > self.sampling_rate:
            self.task_data = self.downsample_data(self.task_data, patient_data.sampling_rate)
        # VOT correction
        self.VOT_correction(break_points)
        # Drop noisy trials
        self.task_data = self.drop_noisy_trials(valid_spectra_points)
        # Create trials
        self.X, self.y = self.create_trials(ecog_data)
        # Print number of trials per class
        self.print_nr_trials()
        # Save trials
        if save_path is not None:
            self.save_trials(save_path)

    def read_taskdata(self, task_path):
        task_file = h5py.File(task_path, 'r')
        task_data = np.array(task_file['task_performance_data'], dtype=np.int32)
        task_file.close()
        return task_data

    def downsample_data(self, data, old_sampling_rate):
        # Take downsampling into account for COT and VOT
        for i in range(data.shape[1]):
            # data[0, i] /= int(old_sampling_rate/self.sampling_rate)
            data[0, i] = int(data[0, i] / (old_sampling_rate/self.sampling_rate))
            if data[1, i] != 0 and data[1, i] != -1:
                data[1, i] = int(data[1, i] / (old_sampling_rate/self.sampling_rate))
        # And for offset
        for i in range(len(self.offset)):
            self.offset[i] = int(self.offset[i] / (old_sampling_rate/self.sampling_rate))
        return data

    def VOT_correction(self, breakpoints):
        for i in range(self.task_data.shape[1]):
            for j in range(len(self.offset)):
                if self.task_data[0, i] < breakpoints[j-1]:
                    current_offset = self.offset[j]
                    break
            if self.task_data[1, i] != 0 and self.task_data[1, i] != -1:
                self.task_data[1, i] += current_offset

    def drop_noisy_trials(self, valid_spectra_points):
        task_data = self.task_data
        raw_trial_length = task_data[0,1]-task_data[0,0]

        # Mark sample points that are 10 st. devs. larger or smaller than the mean signal (over all channels)
        # noisy_samples = np.sum(self.ecog_data, 0) > np.mean(np.sum(self.ecog_data, 0)) + 10*np.std(np.sum(self.ecog_data, 0))
        # noisy_samples += np.sum(self.ecog_data, 0) < np.mean(np.sum(self.ecog_data, 0)) - 10*np.std(np.sum(self.ecog_data, 0))
        # noisy_samples = np.where(noisy_samples==True)
        # for noisy_sample in noisy_samples[0]:
        #     for i in range(task_data.shape[1]):
        #         if task_data[0,i] <= noisy_sample and task_data[0,i]+raw_trial_length >= noisy_sample:
        #             print(f'Dropped noisy trial, index: {i}')
                    # task_data[1,i] = -1+self.offset

        # Drop bad trials indicated with -1
        print(f'Number of bad (-1) trials: {task_data[:, task_data[1, :] == -1].shape[1]}')
        print(f'Number of trials with -1 in column 3: {task_data[:, task_data[2, :] == -1].shape[1]}')
        task_data = task_data[:, task_data[1,:] != -1]
        # Drop too early trials
        valid_points = np.isin(task_data[0,:]+round(self.time_window_start*self.sampling_rate), valid_spectra_points)
        task_data = task_data[:, valid_points]
        # Drop too late trials
        valid_points = np.isin(task_data[0,:]+raw_trial_length, valid_spectra_points)
        task_data = task_data[:, valid_points]
        return task_data

    def create_trials(self, processed_data):
        data, labels = [], []
        startpoint = round(self.time_window_start * self.sampling_rate)
        endpoint = round(self.time_window_stop * self.sampling_rate)
        totalpoints = abs(startpoint)+abs(endpoint)
        # Transpose taskdata to get [trials, info]
        task_data = self.task_data.T
        for trial in task_data:
            # Create trial from VOT if active trial
            if trial[3] in self.label_indices[:-1]:
                # Need to check this because with extracted features shape is [E x B x T]
                # Otherwise (raw or CAR data) just [E x T]
                if len(processed_data.shape) == 3:
                    data.append(processed_data[:, :, trial[1]+startpoint:trial[1]+endpoint])
                else:
                    data.append(processed_data[:, trial[1]+startpoint:trial[1]+endpoint])
                # Store 1 if rest vs. active classifcation, else store phoneme index
                if self.restvsactive:
                    labels.append(1)
                else:
                    labels.append(trial[3])
            # Create trial from QOT if rest trial
            elif trial[3] == self.label_indices[-1]:
                if len(processed_data.shape) == 3:
                    data.append(processed_data[:, :, trial[0]:trial[0]+totalpoints])
                else:
                    data.append(processed_data[:, trial[0]:trial[0]+totalpoints])
                # Store 0 if rest vs. active classifcation, else store phoneme index
                if self.restvsactive:
                    labels.append(0)
                else:
                    labels.append(trial[3])
        data = np.array(data)
        labels = np.array(labels)
        print(f'Shape trials data: {data.shape}')
        return data, labels

    def print_nr_trials(self):
        print('\nNumber of trials')
        for label in np.unique(self.y):
            print(f'Label {label}: {np.count_nonzero(self.y==label)}')
        print(f'Total: {self.y.shape[0]}')
        
    def save_trials(self, save_path):
        # Check if file already exists
        if os.path.exists(save_path):
            print(f'\nOverwriting trials stored in: {save_path}...')
        else:
            print(f'Writing trials to {save_path}...')
        # Store X and y in pickle file
        with open(save_path, 'wb') as f:
            pkl.dump([self.X, self.y], f)