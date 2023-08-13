import numpy as np
import os
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from scipy.signal import cheb2ord
from Net.preprocessing.config import CONSTANT
from sklearn.preprocessing import StandardScaler

CONSTANT = CONSTANT['AutoCar']
raw_path = CONSTANT['raw_path']
n_subjs = 14

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
    time_seg = [[0, 128*2], [128*1, 128*3]]
    # time_seg = [[0, 640]]
    temporal_seg = []
    for [a, b] in time_seg:
        temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis=1))
    """
    (288, 2, 9, 22, 625) 2 represent:self.time_seg = [[0, 625], [375, 1000]]
    """
    temporal_seg = np.concatenate(temporal_seg, axis=1)
    stack_tensor = temporal_seg

    return stack_tensor


def sliding_window(data, window_size, overlap):
    num_channels, num_samples = data.shape
    stride = int(window_size * (1 - overlap))
    num_windows = (num_samples - window_size) // stride + 1

    windows = np.zeros((num_windows, num_channels, window_size))
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows[i, :, :] = data[:, start:end]

    return windows

def get_data():
    sample_rate = 128
    duration = 3  # 秒
    data_points = sample_rate * duration
    labels = []
    train_data = []

    window_size = 128 * duration
    overlap = 0.5
    scaler = StandardScaler()
    for s in range(1, n_subjs + 1):
        folder_path = raw_path + "/Participant " + str(s)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            label = file_path[-5]

            data = np.load(file_path)
            data = scaler.fit_transform(data.T).T
            trimmed_signal = data[:, data_points:-data_points]
            trimmed_signal = trimmed_signal[0:16, 0:1800]
            if trimmed_signal.shape[1] == 1800:
                windows = sliding_window(trimmed_signal, window_size, overlap)
                windows = np.array(windows)
                train_data.append(windows)
                label_num = np.array(train_data).shape[1]
                label_ = np.ones((label_num,))
                label_ *= int(label)
                labels.append(label_)

    train_data = np.concatenate(train_data, axis=0)
    labels = np.array(labels).reshape(-1, )
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, stratify=labels, random_state=123)
    fbank = FilterBank(fs=128, bands= [3, 30], pass_width=4)
    _ = fbank.get_filter_coeff()
    X_train = _tensor_stack(fbank.filter_data(X_train).transpose(1, 0, 2, 3))
    X_test = _tensor_stack(fbank.filter_data(X_test).transpose(1, 0, 2, 3))

    return X_train, X_test, y_train, y_test



