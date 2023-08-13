import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.signal as signal
from scipy.signal import cheb2ord
from Net.preprocessing.BCIC2a import raw
from Net.preprocessing.config import CONSTANT

CONSTANT = CONSTANT['BCIC2a']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
n_trials_per_class = CONSTANT['n_trials_per_class']
n_chs = CONSTANT['n_chs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']
trial_len = CONSTANT['trial_len']


def subject_setting(k_folds, pick_smp_freq, bands, save_path, num_class=4, scenario='CV', sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path_1 = save_path + '/BCIC2a/data_CV_T'
    save_path_2 = save_path + '/BCIC2a/data_CV_E'
    save_path_3 = save_path + '/BCIC2a/data_Holdout'
    n_chs = len(sel_chs)
    n_trials = n_trials_per_class * num_class

    X_train_all, y_train_all = np.zeros((n_subjs, n_trials, n_chs, int(trial_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))
    X_test_all, y_test_all = np.zeros((n_subjs, n_trials, n_chs, int(trial_len * pick_smp_freq))), np.zeros(
        (n_subjs, n_trials))

    id_chosen_chs = raw.chanel_selection(sel_chs)
    for s in range(n_subjs):
        X_train, y_train, X_test, y_test = __load_BCIC2a(raw_path, s + 1, num_class, id_chosen_chs)
        "T.mat"
        X_train_all[s], y_train_all[s] = X_train, y_train
        "E.mat"
        X_test_all[s], y_test_all[s] = X_test, y_test

    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Carry out subject-dependent setting with 10-fold cross validation
    for person, (X_tr, y_tr, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        if len(X_tr.shape) != 3:
            raise Exception('Dimension Error, must have 3 dimension')

        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)

        if scenario == 'CV':
            for fold, (train_index, test_index) in enumerate(skf.split(X_tr, y_tr)):
                print('FOLD:', fold + 1, 'TRAIN:', len(train_index), 'TEST:', len(test_index))
                X_tr_cv, X_te_cv = X_tr[train_index], X_tr[test_index]
                y_tr_cv, y_te_cv = y_tr[train_index], y_tr[test_index]

                fbank = FilterBank(fs=250, bands=bands, pass_width=4)
                _ = fbank.get_filter_coeff()
                '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
                X_tr_fil = _tensor_stack(fbank.filter_data(X_tr_cv, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))
                X_te_fil = _tensor_stack(fbank.filter_data(X_te_cv, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))

                print('Check dimension of training data {} and testing data {}'.format(X_tr_fil.shape, X_te_fil.shape))
                SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person + 1, fold + 1)
                __save_data_with_valset(save_path_1, SAVE_NAME, X_tr_fil, y_tr_cv, X_te_fil, y_te_cv)
                print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person + 1, fold + 1))

            for fold, (train_index, test_index) in enumerate(skf.split(X_te, y_te)):
                print('FOLD:', fold + 1, 'TRAIN:', len(train_index), 'TEST:', len(test_index))
                X_tr_cv, X_te_cv = X_tr[train_index], X_tr[test_index]
                y_tr_cv, y_te_cv = y_tr[train_index], y_tr[test_index]

                fbank = FilterBank(fs=250, bands=bands, pass_width=4)
                _ = fbank.get_filter_coeff()
                '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
                X_tr_fil = _tensor_stack(fbank.filter_data(X_tr_cv, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))
                X_te_fil = _tensor_stack(fbank.filter_data(X_te_cv, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))

                print('Check dimension of training data {} and testing data {}'.format(X_tr_fil.shape, X_te_fil.shape))
                SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person + 1, fold + 1)
                __save_data_with_valset(save_path_2, SAVE_NAME, X_tr_fil, y_tr_cv, X_te_fil, y_te_cv)
                print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person + 1, fold + 1))

        elif scenario == 'Holdout':
            fbank = FilterBank(fs=250, bands=bands, pass_width=4)
            _ = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            X_tr_fil = _tensor_stack(
                fbank.filter_data(X_tr, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))
            X_te_fil = _tensor_stack(
                fbank.filter_data(X_te, window_details={'tmin': 3.0, 'tmax': 7.0}).transpose(1, 0, 2, 3))

            print('Check dimension of training data {} and testing data {}'.format(X_tr_fil.shape, X_te_fil.shape))
            SAVE_NAME = 'Holdout}'
            __save_data_with_valset(save_path_3, SAVE_NAME, X_tr_fil, y_tr, X_te_fil, y_te)


def __load_BCIC2a(PATH, subject, num_class, id_chosen_chs):
    X_train, y_tr, X_test, y_te = raw.get_data(
        PATH=PATH, subject=subject, num_class=num_class,
        id_chosen_chs=id_chosen_chs)
    return X_train, y_tr, X_test, y_te


def __save_data_with_valset(save_path, NAME, X_train, y_train,  X_test, y_test):
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    print('save DONE')


class FilterBank:
    def __init__(self, fs, bands, pass_width=4, f_width=4):
        self.fs = fs
        self.f_trans = 2
        """
        4Hz ~ 40Hz
        self.f_pass: [ 4  8 12 16 20 24 28 32 36]
        """
        self.f_pass = np.arange(4, 40, pass_width)
        self.f_width = f_width
        self.gpass = bands[0]
        self.gstop = bands[1]
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs / 2  # 125 Hz

        for i, f_low_pass in enumerate(self.f_pass):
            """
            f_pass:
            [ 4 8]
            [ 8 12]
            [12 16]
            [16 20]
            [20 24]
            [24 28]
            [28 32]
            [32 36]
            [36 40]
            """
            f_pass = np.asarray([f_low_pass, f_low_pass + self.f_width])

            """
            f_stop:
            [ 2 10]
            [ 6 14]
            [10 18]
            [14 22]
            [18 26]
            [22 30]
            [26 34]
            [30 38]
            [34 42]
            """
            f_stop = np.asarray([f_pass[0] - self.f_trans, f_pass[1] + self.f_trans])

            wp = f_pass / Nyquist_freq
            ws = f_stop / Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i: {'b': b, 'a': a}})

        return self.filter_coeff

    def filter_data(self, eeg_data, window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape

        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin')))
            # +1

        filtered_data = np.zeros((len(self.filter_coeff), n_trials, n_channels, n_samples))

        for i, fb in self.filter_coeff.items():

            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b, a, eeg_data[j, :, :]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:, :, int((window_details.get('tmin')) * self.fs):int(
                    (window_details.get('tmax')) * self.fs)]
            filtered_data[i, :, :, :] = eeg_data_filtered

        return filtered_data


def _tensor_stack(x_fb):
    time_seg = [[0, 625], [375, 1000]]
    temporal_seg = []
    for [a, b] in time_seg:
        temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis=1))
    """
    (288, 2, 9, 22, 625) 2 represent:self.time_seg = [[0, 625], [375, 1000]]
    """
    temporal_seg = np.concatenate(temporal_seg, axis=1)
    stack_tensor = temporal_seg

    return stack_tensor