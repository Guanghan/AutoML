"""
@author: Guanghan Ning
@file: cifar10.py
@time: 10/14/20 11:09 下午
@file_desc: class for Cifar10 dataset.
"""
import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler

from src.dataset.base_dataset import Dataset
from src.dataset.transforms import Compose
from src.core.class_factory import ClassFactory, ClassType
from src.dataset.base_dataset import BaseConfig


class Cifar10CommonConfig(BaseConfig):
    """Default Optim Config."""

    n_class = 10
    batch_size = 256
    num_workers = 8
    train_portion = 1.0
    num_parallel_batches = 64
    fp16 = False


class Cifar10TrainConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='RandomCrop', size=32, padding=4),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        # rgb_mean = np.mean(train_data, axis=(0, 1, 2))/255
        # rgb_std = np.std(train_data, axis=(0, 1, 2))/255
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    padding = 8
    num_images = 50000


class Cifar10ValConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000
    num_images_train = 50000


class Cifar10TestConfig(Cifar10CommonConfig):
    """Default Cifar10 config."""

    transforms = [
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 10000


class Cifar10Config(object):
    """Default Dataset config for Cifar10."""

    common = Cifar10CommonConfig
    train = Cifar10TrainConfig
    val = Cifar10ValConfig
    test = Cifar10TestConfig


@ClassFactory.register(ClassType.DATASET)
class Cifar10(CIFAR10, Dataset):
    """This is a class for Cifar10 dataset.

    Arguments:
        mode (str, optional): `train`,`val` or `test`, defaults to `train`
        cfg (yml, py or dict): the config the dataset need, defaults to None, and if the cfg is None,
             the default config will be used, the default config file is a yml file with the same name of the class
    """

    config = Cifar10Config()

    def __init__(self, **kwargs):
        """Construct the Cifar10 class."""
        Dataset.__init__(self, **kwargs)
        CIFAR10.__init__(self, root=self.args.data_path, train=self.train,
                         transform=Compose(self.transforms.__transform__), download=self.args.download)

    @property
    def input_channels(self):
        """Input channel number of the cifar10 image.

        Return:
            the channel number (int)
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of cifar10 image.

        Return:
            the input size (int)
        """
        _shape = self.data.shape
        return _shape[1]

    def _init_sampler(self):
        """Init sampler used to cifar10.
           raises ValueError: the mode should be train, val or test, if not, will raise ValueError

        Return:
            a sampler method
            if the mode is test, return None, else return a sampler object
        """
        if self.mode == 'test' or self.args.train_portion == 1:
            return None
        self.args.shuffle = False
        num_train = 50000
        indices = list(range(num_train))
        split = int(np.floor(self.args.train_portion * num_train))
        if self.mode == 'train':
            return SubsetRandomSampler(indices[:split])
        elif self.mode == 'val':
            return SubsetRandomSampler(indices[split:num_train])
        else:
            raise ValueError('the mode should be train, val or test')
