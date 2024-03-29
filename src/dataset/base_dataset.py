"""
@author: Guanghan Ning
@file: base_dataset.py
@time: 10/14/20 9:16
@file_desc: Base class of the dataset.
"""
import importlib
from torch.utils import data as torch_data
from src.core.base_task import Task
from src.dataset.transforms import Transforms
from src.core.class_factory import ClassFactory, ClassType
from src.utils.utils_cfg import Config, class2config
from src.utils.utils_dict import update_dict


class Dataset(Task):
    """This is the base class of the dataset, which is a subclass of `TaskOps`.

    The Dataset provide several basic attribute like dataloader, transform and sampler.
    """

    def __new__(cls, *args, **kwargs):
        """Create a subclass instance of dataset."""
        if Dataset in cls.__bases__:
            return super().__new__(cls)
        if kwargs.get('type'):
            t_cls = ClassFactory.get_cls(ClassType.DATASET, kwargs.pop('type'))
        else:
            t_cls = ClassFactory.get_cls(ClassType.DATASET)
        return super().__new__(t_cls)

    def __init__(self, hps=None, mode='train', **kwargs):
        """Construct method."""
        super(Dataset, self).__init__()
        self.args = dict()
        self.mode = mode
        if mode == "val" and not hasattr(self.config, "val"):
            self.mode = "test"
        # modify config from kwargs, `Cifar10(mode='test', data_path='/cache/datasets')`
        if kwargs:
            self.args = Config(kwargs)
        if hasattr(self, 'config'):
            config = Config()
            config = class2config(config_dst=config, class_src=getattr(self.config, self.mode))
            config.update(self.args)
            self.args = config
        self._init_hps(hps)
        self.train = self.mode in ["train", "val"]
        transforms_list = self._init_transforms()
        self._transforms = Transforms(transforms_list)
        if "transforms" in kwargs.keys():
            self._transforms.__transform__ = kwargs["transforms"]
        self.dataset_init()
        self.sampler = self._init_sampler()
        print("Using dataset sampler: {}".format(self.sampler))

    def dataset_init(self):
        """Init Dataset before sampler."""
        pass

    def _init_hps(self, hps):
        """Convert trainer values in hps to cfg."""
        if hps is not None:
            self.args = Config(update_dict(hps, self.args))

    @property
    def dataloader(self):
        """Dataloader attribute which is a unified interface to generate the data.

        Return:
            a batch of data (dict, list, optional)
        """
        print("Shuffle: {}".format(self.args["shuffle"]))
        data_loader = torch_data.DataLoader(self,
                                            batch_size=self.args["batch_size"],
                                            shuffle=self.args["shuffle"],
                                            num_workers=self.args["num_workers"],
                                            pin_memory=self.args["pin_memory"],
                                            sampler=self.sampler,
                                            drop_last=self.args["drop_last"])
        return data_loader

    @property
    def transforms(self):
        """Transform function which can replace transforms."""
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        """Set function of transforms."""
        if isinstance(value, list):
            self.transforms.__transform__ = value

    def _init_transforms(self):
        """Initialize transforms method.

        Return:
            a list of objects
        """
        if "transforms" in self.args.keys():
            transforms = list()
            if not isinstance(self.args["transforms"], list):
                self.args["transforms"] = [self.args["transforms"]]
            for i in range(len(self.args["transforms"])):
                transform_name = self.args["transforms"][i].pop("type")
                kwargs = self.args["transforms"][i]
                if ClassFactory.is_exists(ClassType.TRANSFORM, transform_name):
                    transforms.append(ClassFactory.get_cls(ClassType.TRANSFORM, transform_name)(**kwargs))
                else:
                    transforms.append(getattr(importlib.import_module('torchvision.transforms'),
                                              transform_name)(**kwargs))
            return transforms
        else:
            return list()

    @property
    def sampler(self):
        """Sampler function which can replace sampler."""
        return self._sampler

    @sampler.setter
    def sampler(self, value):
        """Set function of sampler."""
        self._sampler = value

    def _init_sampler(self):
        """Initialize sampler method.

        Return:
            if the distributed is True, return a sampler object, else return None
        """
        sampler = None
        return sampler

    def __len__(self):
        """Get the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get an item of the dataset according to the index."""
        raise NotImplementedError
