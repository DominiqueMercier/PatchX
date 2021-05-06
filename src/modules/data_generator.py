import numpy as np
import tensorflow as tf

from modules.patch_generator import get_patch, get_patch_params


class DataGenerator(tf.keras.utils.Sequence):
    """Dataloader for the patches. Uses a dataset plus additional parameters to refer to the patches.
    """

    def __init__(self, list_IDs, data, labels, dim, config, zero=False, attach=True, notemp=False, batch_size=32, shuffle=True):
        """Init method for the dataloader.

        Args:
            list_IDs (arr): [description]
            data (arr): dataset
            labels (arr): sparse labels
            dim (list): sequence length, channels
            config (list): stride, len pairs for patches
            zero (bool, optional): flag to set data outside the patch to zero. Defaults to False.
            notemp (bool, optional): flag to remove the temporal axis. Defaults to False.
            batch_size (int, optional): batch size. Defaults to 32.
            shuffle (bool, optional): flag to shuffle the data. Defaults to True.
        """
        self.dim = np.copy(dim)
        if attach:
            self.dim[-1] += 1
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.list_IDs = list_IDs
        self.config = config
        self.zero = zero
        self.attach = attach
        self.notemp = notemp
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, dim)
        # Initialization
        bs = len(list_IDs_temp)
        X = np.empty((bs, *self.dim))
        y = np.empty((bs), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # patch params
            sidx, start, end = get_patch_params(
                ID, len(self.labels), self.dim[0], self.config)

            # Store sample
            X[i, ] = get_patch(self.data[sidx], start,
                               end, self.zero, self.attach, self.notemp)

            # Store class)
            y[i] = self.labels[sidx]

        return X, y

    def switch_shuffle(self, switch):
        self.shuffle = switch
        self.on_epoch_end()


class DataGenerator_sample(tf.keras.utils.Sequence):
    """Generic dataloader for samples.
    """

    def __init__(self, list_IDs, data, labels, dim, batch_size=32, shuffle=True):
        """Init method for the dataloader.

        Args:
            list_IDs (arr): [description]
            data (arr): dataset
            labels (arr): sparse labels
            batch_size (int, optional): batch size. Defaults to 32.
            shuffle (bool, optional): flag to shuffle the data. Defaults to True.
        """
        self.dim = np.copy(dim)
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, dim)
        # Initialization
        bs = len(list_IDs_temp)
        X = np.empty((bs, *self.dim))
        y = np.empty((bs), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            X[i, ] = self.data[ID]

            # Store class
            y[i] = self.labels[ID]

        return X, y

    def switch_shuffle(self, switch):
        self.shuffle = switch
        self.on_epoch_end()
