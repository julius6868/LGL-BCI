import numpy as np
import scipy.io as sio
from scipy import signal
from Net.preprocessing.config import CONSTANT

CONSTANT = CONSTANT['BCIC2a']
n_chs = CONSTANT['n_chs']
n_trials = 2*CONSTANT['n_trials']
window_len = CONSTANT['trial_len']*CONSTANT['orig_smp_freq']
orig_chs = CONSTANT['orig_chs']
trial_len = CONSTANT['trial_len'] 
orig_smp_freq = CONSTANT['orig_smp_freq']


def read_raw(PATH, subject, training, num_class, id_chosen_chs):
    data = np.zeros((n_trials, n_chs, window_len))
    label = np.zeros(n_trials)

    NO_valid_trial = 0
    if training:
        mat = sio.loadmat(PATH+'/A0'+str(subject)+'T.mat')['data']
    else:
        mat = sio.loadmat(PATH+'/A0'+str(subject)+'E.mat')['data']
    for ii in range(0,mat.size):
        mat_1 = mat[0,ii]
        mat_2 = [mat_1[0,0]]
        mat_info = mat_2[0]
        _X = mat_info[0]
        _trial = mat_info[1]
        _y = mat_info[2]
        _fs = mat_info[3]
        _classes = mat_info[4]
        _artifacts = mat_info[5]
        _gender = mat_info[6]
        _age = mat_info[7]
        for trial in range(0,_trial.size):
            if(_y[trial][0] <= num_class): 
                data[NO_valid_trial,:,:] = np.transpose(_X[int(_trial[trial]):(int(_trial[trial])+window_len), :20])
                # selected merely motor cortices region
                label[NO_valid_trial] = int(_y[trial])
                NO_valid_trial +=1
    data = data[0:NO_valid_trial, id_chosen_chs, :]
    label = label[0:NO_valid_trial]-1 # -1 to adjust the values of class to be in range 0 and 1
    return data, label


def chanel_selection(sel_chs): 
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id


def get_data(PATH, subject, num_class, id_chosen_chs):
    X_train, y_tr = read_raw(PATH=PATH, subject=subject,
                             training=True, num_class=num_class, id_chosen_chs=id_chosen_chs)
    X_test, y_te = read_raw(PATH=PATH, subject=subject,
                            training=False, num_class=num_class, id_chosen_chs=id_chosen_chs)
    print("Verify dimension training {} and testing {}".format(X_train.shape, X_test.shape))
    return X_train, y_tr, X_test, y_te


def resampling(data, new_smp_freq, data_len):
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*new_smp_freq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i, j, :] = signal.resample(data[i, j, :], new_smp_point)
    return data_resampled