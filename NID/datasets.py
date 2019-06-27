from .utils import data_augmentation
import numpy as np
import scipy.io as sio
import os
import h5py
import random
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, makedir_exist_ok
import pdb


class Pinky40(Dataset):
    """ pinky40 EM footprints
    Args:
        root (string): Root directory of all datasets

        train (bool, option): If True, create dataset from ``training_x.h5'',
            otherwise from ``val.h5''

        patch (integer): patch size for croping the training images

        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.

    """

    urls = ['http://www.columbia.edu/~pz2230/sharing/datasets/Pinky40/EM_footprints.mat']

    def __init__(self, root=None, train=True, patch=40,
                 stride=10, download=True):
        super(Pinky40, self).__init__()
        if root is None:
            import inspect
            self.root = os.path.abspath(os.path.join(
                os.path.dirname(inspect.getfile(Pinky40)),
                '..', 'datasets'))
        else:
            self.root = root

        self.patch = patch
        self.stride = stride
        self.train = train

        # download the file
        if download:
            self.download

        # check the training and testing data
        if not self._check_exists_processed():
            if self._check_exist_raw():
                # process data
                self.process_raw()
            else:
                # report an error
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')

        if self.train:
            h5f = h5py.File(self.training_file, "r")
        else:
            h5f = h5py.File(self.validating_file, "r")

        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return os.path.join(self.processed_folder,
                            'train_{}_{}.h5'.format(self.patch,
                                                    self.stride))

    @property
    def validating_file(self):
        return os.path.join(self.processed_folder,
                            'valid.h5')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.training_file, "r")
        else:
            h5f = h5py.File(self.validating_file, "r")
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

    def _check_exists_processed(self):
        return (os.path.exists(self.training_file)
                and os.path.exists(self.validating_file))

    def _check_exist_raw(self):
        for url in self.urls:
            file_path = os.path.join(self.raw_folder, os.path.basename(url))
            if not os.path.exists(file_path):
                return False
        return True

    def download(self):
        """Download the Pinky40 data if it doesn't exist in the data folder"""
        # create a folder for storing the raw data
        makedir_exist_ok(self.raw_folder)

        # download files
        for url in self.urls:
            file_path = os.path.join(self.raw_folder, os.path.basename(url))
            if not os.path.exists(file_path):
                download_url(url, self.raw_folder)

    def process_raw(self, create_new=False, aug_times=1):
       # check whether it's necessary to create new
        if self._check_exists_processed() and (not create_new):
            print('the training and testing data has been created already')
            return

        # create new
        print('creating  the training and validating dataset...')
        patch_size = self.patch
        file_path = os.path.join(
            self.raw_folder, 'EM_footprints.mat')  # raw data file
        if not os.path.exists(file_path):
            self.download()
        imgs = sio.loadmat(file_path)['data']
        h, w, frames = imgs.shape

        # split data into training set and testing set
        makedir_exist_ok(self.processed_folder)
        ind_train = (np.random.rand(frames) > 0.1)
        imgs_train = imgs[:, :, ind_train]
        imgs_val = imgs[:, :, np.logical_not(ind_train)]

        # create training data
        if os.path.exists(self.training_file):
            os.remove(self.training_file)
        h5f = h5py.File(self.training_file, 'w')
        scales = [2, 1.5, 1, 0.8]
        train_num = 0
        for i in range(imgs_train.shape[-1]):
            img = imgs_train[:, :, i]
            img = img / np.max(img)  # normalize data by its maximum value
            for k in range(len(scales)):
                Img = cv2.resize(
                    img, (int(w*scales[k]), int(h*scales[k])), interpolation=cv2.INTER_CUBIC)
                Img = np.expand_dims(Img[:, :].copy(), 0)
                Img = np.float32(Img)
                patches = Im2Patch(Img, win=patch_size, stride=self.stride)
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset('{}_{}_{}'.format(i, scales[k], n),
                                       data=data)
                    train_num += 1
                    for m in range(aug_times-1):
                        data_aug = data_augmentation(
                            data, np.random.randint(1, 8)).flatten()
                        h5f.create_dataset('{}_{}_{}_aug_{}'.format(
                            i, scales[k], n, m), data=data_aug)
                        train_num += 1
            if i % 100 == 99:
                print('processed {0}/{1} images: {2} training examples'.
                      format(i+1, imgs_train.shape[-1], train_num))
        h5f.close()

        # testing data
        print('\nprocess validation data')
        if os.path.exists(self.validating_file):
            os.remove(self.validating_file)
        h5f = h5py.File(self.validating_file, 'w')
        val_num = 0
        for i in range(imgs_val.shape[-1]):
            img = imgs_val[:, :, i]
            img = np.expand_dims(img[:, :], 0) / img.max()
            img = np.float32(img)
            h5f.create_dataset(str(val_num), data=img)
            val_num += 1
        h5f.close()

        # summarize results
        print('training set, # samples %d\n' % train_num)
        print('val set, # samples %d\n' % val_num)


def Im2Patch(img, win, stride=1):
    """create multiple patches by croping 1 image

    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    if endw < win:
        img[:, endw:win, :] = 0
        endw = img
    if endh < win:
        img[:, :, endh:win] = 0
        endh = win
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    Y = Y[:, :, Y.max(axis=1).max(axis=0) > 0.1]  # remove small components
    return Y.reshape([endc, win, win, -1])
