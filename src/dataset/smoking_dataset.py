"""
@author: Guanghan Ning
@file: Smoking_dataset.py
@time: 10/14/20 11:09
@file_desc: class for Smoking dataset.
"""
import numpy as np
import torch

from src.dataset.smoking import Smoking
from torch.utils.data.sampler import SubsetRandomSampler

from src.dataset.base_dataset import Dataset
from src.dataset.transforms import Compose
from src.core.class_factory import ClassFactory, ClassType
from src.core.default_config import BaseConfig


class SmokingCommonConfig(BaseConfig):
    """Default Optim Config."""
    n_class = 2
    batch_size = 8
    num_workers = 4
    train_portion = 1.0
    num_parallel_batches = 8
    fp16 = False


class SmokingTrainConfig(SmokingCommonConfig):
    """Default Smoking config."""

    transforms = [
        #dict(type='RandomCrop', size=128, padding=4),
        #dict(type='RandomCrop', size=224, padding=4),
        #dict(type='RandomResizedCrop', size=224, padding=4),
        dict(type='RandomHorizontalFlip'),
        dict(type='RandomPerspective', distortion_scale=0.3, p=0.5, interpolation=3),
        #    GaussianBlur([.0, 1.])
        dict(type='ToTensor'),
        # rgb_mean = np.mean(train_data, axis=(0, 1, 2))/255
        # rgb_std = np.std(train_data, axis=(0, 1, 2))/255
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        #dict(type='Normalize', mean=rgb_mean, std=rgb_std)
        ]
    padding = 8
    num_images = 11669+154620


class SmokingValConfig(SmokingCommonConfig):
    """Default Smoking config."""

    transforms = [
        #dict(type='Resize', size=128),
        dict(type='Resize', size=224),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 1460+19327
    num_images_train = 11669+154620


class SmokingTestConfig(SmokingCommonConfig):
    """Default Smoking config."""

    transforms = [
        #dict(type='Resize', size=128),
        dict(type='Resize', size=224),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])]
    num_images = 1460+19327


class SmokingConfig(object):
    """Default Dataset config for Smoking."""

    common = SmokingCommonConfig
    train = SmokingTrainConfig
    val = SmokingValConfig
    test = SmokingTestConfig


@ClassFactory.register(ClassType.DATASET)
class SmokingDataset(Smoking, Dataset):
    """This is a class for Smoking dataset.

    Arguments:
        mode (str, optional): `train`,`val` or `test`, defaults to `train`
        cfg (yml, py or dict): the config the dataset need, defaults to None, and if the cfg is None,
             the default config will be used, the default config file is a yml file with the same name of the class
    """

    config = SmokingConfig()

    def __init__(self, **kwargs):
        """Construct the Smoking class."""
        Dataset.__init__(self, **kwargs)
        Smoking.__init__(self,
                         root=self.args["data_path"],
                         split=self.mode,
                         transform=Compose(self.transforms.__transform__))

    @property
    def input_channels(self):
        """Input channel number of the image.

        Return:
            the channel number (int)
        """
        _shape = self.data.shape
        _input_channels = 3 if len(_shape) == 4 else 1
        return _input_channels

    @property
    def input_size(self):
        """Input size of image.

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
        print("Initiating dataset sampler, mode={}".format(self.mode))
        if self.mode == 'test' or self.mode == 'val':
            return None
        self.args["shuffle"] = False
        num_train = 11669+154620
        num_val = 1460+19327
        if self.mode == 'train':
            neg_samples_weights = [10.0/154620 for i in range(154620)]
            #print("len(neg_sample_weights): {}".format(len(neg_samples_weights)))
            pos_samples_weights = [3.0/11669 for i in range(11669)]
            #print("len(pos_sample_weights): {}".format(len(pos_samples_weights)))
            neg_samples_weights.extend(pos_samples_weights)
            samples_weights = torch.tensor(neg_samples_weights, dtype=torch.float)
            #print("len(sample_weights): {}".format(samples_weights))
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True)
            return sampler
        else:
            raise ValueError('the mode should be train, val or test')
