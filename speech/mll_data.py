from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import numpy as np

from generate_speech_data import SENTENCES_MLL_PATH


class MllData(object):
    """The multi-label learning input data."""

    @property
    def num_steps(self):
        return self.max_steps

    @property
    def num_classes(self):
        return self.n_classes

    def __init__(self, raw_input, raw_labels, cell_size):
        '''Object with input data for LSTM-MLL NN
             raw_input: list of 2D numpy arrays with raw mfcc input frames [timesteps x n_features]
             raw_labels: list of 1D arrays with labels [n_classes]
             cell_size: int with the size on lstm cell

             WARNING: cell_size should be multiple of n_features
        '''

        assert len(raw_input) == len(raw_labels), "input len %d != labels len %d" % (len(raw_input), len(labels))

        self.input_size = len(raw_labels)
        self.raw_input = raw_input
        self.raw_labels = raw_labels
        self.cell_size = cell_size

        self.n_features = raw_input[0].shape[1]
        self.n_classes = raw_labels[0].shape[0]
        assert cell_size % self.n_features == 0, "cell size should be multiple num of features"
        cell_size_factor = cell_size // self.n_features

        self.max_timesteps = 0
        for input_index in range(len(raw_labels)):
            self.max_timesteps = max(self.max_timesteps, raw_input[input_index].shape[0])
        print("max timesteps", self.max_timesteps)
        self.max_timesteps += cell_size_factor - self.max_timesteps % cell_size_factor
        print("increased max timesteps", self.max_timesteps)
        self.max_steps = self.max_timesteps // cell_size_factor
        print("max steps", self.max_steps)

    def get_batch(self, batch_size):
        '''Produce random batch from raw input data
             batch_size: int with number of inputs/labels per batch
             returns: batch tuple (inputs, labels) consists of
                    inputs = 3D array w/ shape [batch_size x max_steps x cell_size]
                    labels = 2D array w/ shape [batch_size x n_classes]
        '''
        random_indexes = np.random.permutation(self.input_size)

        inputs = np.zeros([batch_size, self.max_steps, self.cell_size])
        labels = np.zeros([batch_size, self.n_classes])
        for batch_index, raw_index in enumerate(random_indexes[0:batch_size]):
            mfcc = self.raw_input[raw_index]
            # pad with zeros to max_timesteps
            pad_len = self.max_timesteps - mfcc.shape[0]
            padded = np.pad(mfcc, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
            # reshape time_steps x n_features -> steps x cell_size
            inputs[batch_index] = padded.reshape([self.max_steps, self.cell_size])
            labels[batch_index] = self.raw_labels[raw_index]
        return inputs, labels


def load_data(path, swap_axes=True):
    print("load data from " + path)
    texts = {}
    types = {}
    for line in open(path + "/lines.list").readlines():
        num, type, text = line.split(":")
        types[num] = type
        texts[num] = text.replace("\n", '')
    train = {'texts': [], 'mfcc': [], 'labels': []}
    validation = {'texts': [], 'mfcc': [], 'labels': []}
    for file_name in os.listdir(path + "/mfcc/"):
        num, voice, rate = file_name.split("_")
        if types[num] == "train":
            target = train
        else:
            target = validation
        target['texts'].append(texts[num])
        mfcc = np.load(os.path.join(path + "/mfcc/", file_name))
        if swap_axes:
            mfcc = np.swapaxes(mfcc, 0, 1)
        target['mfcc'].append(mfcc)
        target['labels'].append(np.load(os.path.join(path + "/labels/", num + ".npy")))
    return train, validation


def main():
    train, validation = load_data(SENTENCES_MLL_PATH)
    validation_input = MllData(validation['mfcc'], validation['labels'], 208)
    v_inputs, v_labels = validation_input.get_batch(5)


if __name__ == '__main__':
    main()
    print("DONE!")