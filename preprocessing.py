import numpy as np
import matplotlib.pyplot as plt
import pycwt
import h5py
from tqdm import tqdm
import scipy.signal
from mne.filter import notch_filter, filter_data

class Preprocessor():
    def __init__(self, 
                 data_filename,
                 patient_data,
                 sampling_rate=512,
                 buffer=64,
                 preprocessing_type='raw'):
        
        self.patient_data = patient_data
        self.sampling_rate = sampling_rate
        self.buffer = buffer # Samples at beginning and end of file to exclude
        self.preprocessing_type = preprocessing_type

        if self.preprocessing_type == 'given_features':
            self.ecog_params, self.dataStructure, self.data_params, self.ecog_data, self.valid_spectra_pnts = self.read_preprocessed_data(data_filename)
            # To avoid errors caused by data_params[break_pnts] either being a float or an array of floats
            if np.prod(self.data_params['break_pnts'].shape) > 1:
                self.break_points = [0]
                for break_point in self.data_params['break_pnts']:
                    self.break_points.append(int(break_point))
                self.break_points.append(self.ecog_data.shape[1])
            else:
                self.break_points = [0, int(self.data_params['break_pnts']), self.ecog_data.shape[1]]
        # Manually preprocess
        else:
            self.HDecog_channels = patient_data.HD_channels
            self.n_channels = 0
            for channel_set in self.HDecog_channels:
                self.n_channels += len(channel_set)
            self.ecog_data, self.valid_spectra_pnts, self.break_points = self.read_raw_data(data_filename)

        print(f'\n\nStart preprocessing for patient \'{patient_data.patient}\'\n')

        # Downsample to self.sampling_rate if needed
        if patient_data.sampling_rate > self.sampling_rate:
            self.ecog_data, self.valid_spectra_pnts, self.break_points = self.downsampling(self.ecog_data, patient_data.sampling_rate)
        
        # Notch filtering
        self.ecog_data = self.apply_filter(self.ecog_data)

        if self.preprocessing_type != 'raw' and self.preprocessing_type != 'given_features':
            # Comman average referencing
            self.ecog_data = self.CAR(self.ecog_data, patient_data.CAR_channels)
            if self.preprocessing_type != 'CAR':
                # Get and plot convolved responses
                self.ecog_data = self.wavelet_transform(self.ecog_data, plot_result=True)
                # Calculate Z-score
                self.ecog_data = self.z_score_spectra(self.ecog_data)
                # Smooth signal
                self.ecog_data = self.smooth_spectra(self.ecog_data)
        
        print(f'ECoG data shape: {self.ecog_data.shape}')
        print(f'Nr of valid spectra points: {len(self.valid_spectra_pnts)}')
        print(f'Break points: {self.break_points}\n')


    def hdf5group_to_dict(self, data_group):
        to_dict = dict()
        for key in data_group:
            item = np.array(data_group[key]).squeeze()
            to_dict[key] = item
        return to_dict

    def read_preprocessed_data(self, data_filename, print_data=True):
        data_file = h5py.File(data_filename, 'r')
        # print(data_file.keys())
        # Convert features in HDF5 Group to Python dictionary
        ecog_params = self.hdf5group_to_dict(data_file['ECoG_feature_params'])
        dataStructure = self.hdf5group_to_dict(data_file['dataStructure'])
        data_params = self.hdf5group_to_dict(data_file['data_params'])
        # filter_params is again an HDF5 Group itself
        data_params['filter_params'] = self.hdf5group_to_dict(data_file['data_params']['filter_params'])
        # Convert data in HDF5 Dataset to numpy arrays
        ecog_data = np.array(data_file['ECoG_features_data'])
        valid_spectra_pnts = np.array(data_file['valid_spectra_pnts'])
        data_file.close()
        # Print data features
        if print_data:
            print(f'\nECoG_params:\n{ecog_params.keys()}')
            print(f'\ndataStructure:\n{dataStructure.keys()}')
            print(f'\ndata_params:\n{data_params.keys()}')
            print(f'\necog_data:\n{ecog_data.shape}')
            print(f'\nvalid_spectra_pnts:\n{valid_spectra_pnts.shape}')
        return ecog_params, dataStructure, data_params, ecog_data, valid_spectra_pnts

    def read_raw_data(self, data_filenames):
        # Read the files in data_filenames
        data_dicts = []
        for datafile in data_filenames:
            data_dicts.append(self.hdf5group_to_dict(h5py.File(datafile, 'r')))

        # Define valid points
        # for each file don't include first and last x points, with x=buffer
        valid_points = []
        begin_point = 0
        for data_i in range(len(data_dicts)):
            n_samples = data_dicts[data_i]['ECoG_data'].shape[1]
            for j in range(begin_point+self.buffer, begin_point+n_samples-self.buffer):
                valid_points.append(j)
            begin_point += n_samples

        # Concatenate data from different files
        ecog_data = np.empty([self.n_channels,0])
        for data_i, data_dict in enumerate(data_dicts):
            new_ecog_data = np.array(data_dict['ECoG_data'][self.HDecog_channels[0]])
            for i in range(1, len(self.HDecog_channels)):
                new_ecog_data = np.vstack((new_ecog_data, data_dict['ECoG_data'][self.HDecog_channels[i]]))
            ecog_data = np.hstack((ecog_data, new_ecog_data))

        # Define breakpoints (points that split files)
        break_points = [0]
        point = 0
        for i in range(len(data_dicts)):
            point += data_dicts[i]['ECoG_data'].shape[1]
            break_points.append(point)
        return ecog_data, valid_points, break_points

    def downsampling(self, data, old_sampling_rate):
        print(f'Downsampling from {old_sampling_rate}Hz to {self.sampling_rate}Hz...')
        downsampled_data = []
        n_samples = int(data.shape[1] / (old_sampling_rate/self.sampling_rate))
        for channel in tqdm(range(data.shape[0])):
            downsampled_data.append(scipy.signal.resample(data[channel], n_samples))
        # Also correctly downsample break- and valid points
        breakpoints = [int(self.break_points[i]/(old_sampling_rate/self.sampling_rate)) for i in range(len(self.break_points))]
        valid_points = []
        for i in range(len(breakpoints)-1):
            for j in range(breakpoints[i]+self.buffer, breakpoints[i+1]-self.buffer):
                valid_points.append(j)

        return np.array(downsampled_data), valid_points, breakpoints

    def apply_filter(self, data):
        print(f'Notch filtering...')
        freqs = np.array([50, 100])
        data = notch_filter(x=data, Fs=self.sampling_rate, freqs=freqs, verbose=True)
        if self.patient_data.ID == '7.2':
            data = filter_data(data=data, sfreq=self.sampling_rate, l_freq=150, h_freq=110, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, verbose=True)
        return data

    def CAR(self, data, CAR_channels):
        # Comman Average Referencing
        print(f'Common average referencing...')
        # Use all channels
        if CAR_channels is None:
            car_data = data - np.mean(data, 0)
        # Use specified set of channels
        else:
            car_data = data - np.mean(data[CAR_channels], 0)
        return car_data

    # TODO: find better way to include RvA
    # TODO: add more comments
    def wavelet_transform(self, data, plot_result=False, nr_channels=-1):
        if nr_channels == -1:
            nr_channels = data.shape[0]
        sampling_period = 1/self.sampling_rate

        freq_bands = {
            'delta': np.arange(1,5),
            'theta': np.arange(4,9),
            'alpha': np.arange(8,13),
            'beta': np.arange(13,31),
            # 'lowgamma': np.arange(30,71),
            # 'highgamma': np.arange(70,151),
            'gamma': np.arange(40,151), # Previously called 'broadband40-150'
            'gestures_gamma': np.arange(70,126),
            'gestures_gamma_2': np.arange(70,126)
        }
        if self.patient_data.ID == '7.2':
            freq_bands['gamma'] = np.arange(40,111)

        if self.preprocessing_type == 'allbands':
            freqs = [freq_bands['delta'], freq_bands['theta'], freq_bands['alpha'], freq_bands['beta'], freq_bands['gamma']]
        elif self.preprocessing_type == 'articleHFB':
            freqs = [np.arange(65,126)]
        else:
            freqs = [freq_bands[self.preprocessing_type]]
            
        # print(data.shape)
        print(f'Calculating wavelet transform for \'{self.preprocessing_type}\' for {nr_channels} channels...')
        response_spectra = []
        for i in tqdm(range(nr_channels)):
            # print(f'Channel {i+1}...')
            single_electrode_responses = []
            for freqsubset in freqs:
                # print(f'freqsubset: {freqsubset}')
                response, _, _, _, _, _ = pycwt.cwt(data[i], sampling_period, wavelet='morlet', freqs=freqsubset)
                # print(np.array(response).shape)
                # Take real part of spectra and log(amplitude)
                norm_spectra = abs(response)
                norm_spectra[np.isnan(norm_spectra)] = 0
                norm_spectra = np.log(norm_spectra)
                single_electrode_responses.append(np.mean(np.array(norm_spectra), 0))

            response_spectra.append(single_electrode_responses)
        '''
        if plot_result:    
            plt.matshow(gamma_feature, aspect='auto')
            plt.show()
        '''
        return np.array(response_spectra)

    def z_score_spectra(self, spectra):
        print(f'Z-scoring spectra...')
        # Spectra: E x B x T
        # buffer = 64 # Original code: (2*ECoG_feature_params.span*ceil(ECoG_feature_params.sample_rate/ECoG_feature_params.spectra(1)));
                    # But parameter span not used here (because default used for Morlet wavelet)
        for freq_band in tqdm(range(spectra.shape[1])):
            for i in range(len(self.break_points)-1):
                chunk_points = np.arange(self.break_points[i]+self.buffer, self.break_points[i+1]-self.buffer)
                chunk = spectra[:, freq_band, chunk_points] 
                # Calculate Z-score over chunk
                spectra[:, freq_band, np.arange(self.break_points[i], self.break_points[i+1])] = \
                    (spectra[:, freq_band, np.arange(self.break_points[i], self.break_points[i+1])] - np.mean(chunk)) / np.std(chunk)
        return spectra

    def smooth_spectra(self, data):
        # Smoothing
        if self.preprocessing_type == 'gestures_gamma':
            smooth_window = 0.5 # seconds
        else:
            smooth_window = 0.1 # seconds
        smoothkernel_size = round(smooth_window*self.sampling_rate)
        kernel = np.ones(smoothkernel_size)
        print(f'Smoothing spectra (with kernel size {smoothkernel_size})...')
        smooth_norm_spectra = []
        for channel in tqdm(range(data.shape[0])):
            single_channel_spectra = []
            for freq_band in range(data.shape[1]):
                single_channel_spectra.append(np.convolve(data[channel, freq_band], kernel, 'same')/smoothkernel_size)
            smooth_norm_spectra.append(np.array(single_channel_spectra))
        smooth_norm_spectra = np.array(smooth_norm_spectra)
        return smooth_norm_spectra

