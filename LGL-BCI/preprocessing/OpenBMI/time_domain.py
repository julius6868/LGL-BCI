import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from Net.utils import butter_bandpass_filter
import scipy.signal as signal
from scipy.signal import cheb2ord
from Net.preprocessing.OpenBMI import raw
from Net.preprocessing.config import CONSTANT
CONSTANT = CONSTANT['OpenBMI']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

def subject_dependent_setting(k_folds, pick_smp_freq, bands, order, save_path, num_class=2, sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path = save_path + '/OpenBMI/time_domain/{}_class/subject_dependent'.format(num_class)

    id_chosen_chs = raw.chanel_selection(sel_chs)
    X_train_all, y_train_all, X_test_all, y_test_all = __load_OpenBMI(raw_path, n_subjs, pick_smp_freq, num_class, MI_len, id_chosen_chs)
    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Carry out subject-dependent setting with k-fold cross validation
    for person, (X_tr, y_tr, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        if len(X_tr.shape) != 3:
            raise Exception('Dimension Error, must have 3 dimension')

        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr , y_tr)):
            print('FOLD:', fold+1, 'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))
            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]
            print("111111111111111111", X_tr_cv.shape)

            print('Band-pass filtering from {} to {} Hz.'.format(bands[0],  bands[1]))
            fbank = FilterBank(fs=250, bands=bands, pass_width=4)
            _ = fbank.get_filter_coeff()
            X_tr_fil = _tensor_stack(fbank.filter_data(X_tr_cv).transpose(1, 0, 2, 3))
            print("333333333333333", X_tr_fil.shape)
            X_val_fil = _tensor_stack(fbank.filter_data(X_val_cv).transpose(1, 0, 2, 3))
            X_te_fil = _tensor_stack(fbank.filter_data(X_te).transpose(1, 0, 2, 3))
            print('Check dimension of training data {}, val data {} and testing data {}'.format(X_tr_fil.shape, X_val_fil.shape, X_te_fil.shape))
            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person+1, fold+1)
            __save_data_with_valset(save_path, SAVE_NAME, X_tr_fil, y_tr_cv, X_val_fil, y_val_cv, X_te_fil, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person+1, fold+1))
                      
def subject_independent_setting(k_folds, pick_smp_freq, bands, order, save_path, num_class=2, sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path = save_path + '/OpenBMI/time_domain/{}_class/subject_independent'.format(num_class)

    id_chosen_chs = raw.chanel_selection(sel_chs)
    X_train_all, y_train_all, X_test_all, y_test_all = __load_OpenBMI(raw_path, n_subjs, pick_smp_freq, num_class, MI_len, id_chosen_chs)
    for directory in [save_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Carry out subject-independent setting with 5-fold cross validation
    for person, (X_val, y_val, X_te, y_te) in enumerate(zip(X_train_all, y_train_all, X_test_all, y_test_all)):
        train_subj = [i for i in range(n_subjs)]
        train_subj = np.delete(train_subj, person) # remove test subject

         # Generating fake data to used for k-fold cross-validation only
        fake_tr = np.zeros((len(train_subj), 2))
        fake_tr_la = np.zeros((len(train_subj)))

        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        for fold, (train_ind, val_ind) in enumerate(skf.split(fake_tr , fake_tr_la)):
            print('FOLD:', fold+1, 'TRAIN:', len(train_ind), 'VALIDATION:', len(val_ind))
            train_index, val_index = train_subj[train_ind], train_subj[val_ind]
            X_train_cat = np.concatenate((X_train_all[train_index], X_test_all[train_index]), axis=0)
            X_val_cat = np.concatenate((X_train_all[val_index], X_test_all[val_index]), axis=0)
            y_train_cat = np.concatenate((y_train_all[train_index], y_test_all[train_index]), axis=0)
            y_val_cat = np.concatenate((y_train_all[val_index], y_test_all[val_index]), axis=0)

            X_train = X_train_cat.reshape(-1, X_train_cat.shape[2], X_train_cat.shape[3])
            y_train = y_train_cat.reshape(-1)
            X_val = X_val_cat.reshape(-1, X_val_cat.shape[2], X_val_cat.shape[3])
            y_val = y_val_cat.reshape(-1)
            X_test = X_te
            y_test = y_te

            # Performing bandpass-filtering
            fbank = FilterBank(fs=250, bands=bands, pass_width=4)
            _ = fbank.get_filter_coeff()
            print('Band-pass filtering from {} to {} Hz.'.format(bands[0],  bands[1]))
            X_train_fil =  _tensor_stack(fbank.filter_data(X_train))
            print("1111111111111111", X_train_fil.shape)
            X_val_fil = _tensor_stack(fbank.filter_data(X_val))
            X_test_fil = _tensor_stack(fbank.filter_data(X_test))

            print('Verify the final dimesion of training data {}, val data {} and testing data {}'.format(X_train_fil.shape, X_val_fil.shape,X_test_fil.shape))
            print("Check class number Tr {} Val {} and Te {}".format(np.unique(y_train),np.unique(y_val),np.unique(y_test)))
            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person+1, fold+1)
            __save_data_with_valset(save_path, SAVE_NAME, X_train_fil, y_train, X_val_fil, y_val, X_test_fil, y_test)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person+1, fold+1))

def __load_OpenBMI(PATH, n_subjs, new_smp_freq, num_class, MI_len, id_chosen_chs):
    X_train_all, y_train_all, X_test_all, y_test_all = raw.load_crop_data(PATH, n_subjs, new_smp_freq, num_class, MI_len, id_chosen_chs)
    return X_train_all, y_train_all, X_test_all, y_test_all

def __save_data_with_valset(save_path, NAME, X_train, y_train, X_val, y_val, X_test, y_test):
    np.save(save_path+'/X_train_'+NAME+'.npy', X_train)
    np.save(save_path+'/X_val_'+NAME+'.npy', X_val)
    np.save(save_path+'/X_test_'+NAME+'.npy', X_test)
    np.save(save_path+'/y_train_'+NAME+'.npy', y_train)
    np.save(save_path+'/y_val_'+NAME+'.npy', y_val)
    np.save(save_path+'/y_test_'+NAME+'.npy', y_test)
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