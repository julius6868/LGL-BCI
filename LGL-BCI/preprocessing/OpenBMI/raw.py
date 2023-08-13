import numpy as np
import scipy.io as sio
from Net.utils import resampling
from Net.preprocessing.config import CONSTANT
from sklearn.model_selection import train_test_split
CONSTANT = CONSTANT['OpenBMI']
raw_path = CONSTANT['raw_path']
orig_chs = CONSTANT['orig_chs']


def read_raw(PATH, subject, training, id_ch_selected):
    if training:
        mat_file_name = PATH + '/sess01'+'_subj'+str(subject).zfill(2)+'_EEG_MI.mat'
        mat = sio.loadmat(mat_file_name)
        print('This is data from: ', mat_file_name)
    else:
        mat_file_name = PATH + '/sess02' + '_subj' + str(subject).zfill(2) + '_EEG_MI.mat'
        mat = sio.loadmat(mat_file_name)
        print('This is data from: ', mat_file_name)

    raw_train_data = mat['EEG_MI_train'][0]['smt'][0]
    raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]
    raw_train_data = np.swapaxes(raw_train_data, 0, 1)
    # print('raw_train_data_shape:', raw_train_data.shape)
    raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
    raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
    raw_test_data = np.swapaxes(raw_test_data, 0, 1)
    # print('raw_test_data_shape:', raw_test_data.shape)
    label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0]-1
    label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0]-1
    return raw_train_data, label_train_data, raw_test_data, label_test_data


def chanel_selection(sel_chs):
    orig_chs = CONSTANT['orig_chs']
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch ,'---', 'Index_is:', ch_id)
    return chs_id


def get_data(PATH, subject, id_chosen_chs):
    X_train_s1, y_tr_s1, X_test_s1, y_te_s1 = read_raw(PATH=PATH, subject=subject,
                                                       training=True, id_ch_selected=id_chosen_chs)
    X_train_s2, y_tr_s2, X_test_s2, y_te_s2 = read_raw(PATH=PATH, subject=subject,
                                                       training=False,  id_ch_selected=id_chosen_chs)
    print([X_train_s1.shape, X_test_s1.shape])
    X_s1 = np.concatenate((X_train_s1, X_test_s1), axis=0)
    y_s1 = np.concatenate((y_tr_s1, y_te_s1), axis=0)
    X_s2 = np.concatenate((X_train_s2, X_test_s2), axis=0)
    y_s2 = np.concatenate((y_tr_s2, y_te_s2), axis=0)
    print("Verify S1 dimension data {} and label {}".format(X_s1.shape, y_s1.shape))
    print("Verify S2 dimension data {} and label {}".format(X_s2.shape, y_s2.shape))
    return  X_s1, y_s1, X_s2, y_s2
