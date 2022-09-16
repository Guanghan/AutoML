"""
@author: Guanghan Ning
@file: transforms.py
@time: 10/14/20 9:22
@file_desc: This is a class for Transforms.
"""
from src.core.class_factory import ClassFactory, ClassType


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        """Construct the Compose class."""
        self.transforms = transforms

    def __call__(self, img):
        """Call function of Compose."""
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        """Construct method."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Compose_pair(object):
    """Composes several transforms together.

    Arguments:
    transforms (callable class): transform method
    """

    def __init__(self, transforms):
        """Construct the Compose_pair class."""
        self.transforms = transforms

    def __call__(self, img1, img2):
        """Call function of Compose_pair.

        Arguments:
            image (PIL Image): usually the feature image, for example, the LR image for super solution dataset,
                               the initial image for the segmentation dataset, etc.
            label (PIL Image): usually the label image, for example, the HR image for super solution dataset,
                               the mask image for the segmentation dataset, etc.

        Return:
            the image after transform (list): every item is a PIL image,
                                              the first one is feature image, the second is label image
        """
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class Transforms(object):
    """This is the base class of the transform.

    The Transforms provide several basic method like append, insert, remove and replace.
    """

    def __init__(self, transform_list=None):
        """Construct Transforms class."""
        self.__transform__ = []
        self._new(transform_list)

    def __call__(self, *args):
        """Call function."""
        if len(args) == 1:
            return Compose(self.__transform__)(*args)
        elif len(args) == 2:
            return Compose_pair(self.__transform__)(*args)
        else:
            raise ValueError("Length of args must be either 1 or 2")

    def _new(self, transform_list):
        """Private method, which generate a list of transform.

        Arguments:
            transform_list (list): a series of transforms
        """
        if isinstance(transform_list, list):
            for trans in transform_list:
                if isinstance(trans, tuple):
                    transform = ClassFactory.get_cls(ClassType.TRANSFORM, trans[0])
                    self.__transform__.append(transform(*trans[1:]))
                elif isinstance(trans, object):
                    self.__transform__.append(trans)
                else:
                    raise ValueError("Unsupported type ({}) to create transforms".format(trans))
        else:
            raise ValueError("Transforms ({}) is not a list".format(transform_list))

    def replace(self, transform_list):
        """Replace the transforms with the new transforms.

        Arguments:
            transform_list (list): a series of transforms
        """
        if isinstance(transform_list, list):
            self.__transform__[:] = []
            self._new(transform_list)

    def append(self, *args, **kwargs):
        """Append a transform to the end of the list.

        Arguments:
           *args (tuple): positional arguments
           ** kwargs (dict): keyword arguments
        """
        if isinstance(args[0], str):
            transform = ClassFactory.get_cls(ClassType.TRANSFORM, args[0])
            self.__transform__.append(transform(**kwargs))
        else:
            self.__transform__.append(args[0])

    def insert(self, index, *args, **kwargs):
        """Insert a transform into the list.

         Arguments:
            index (int): Insertion position
            *args (tuple): positional arguments
            ** kwargs (dict): keyword arguments
        """
        if isinstance(args[0], str):
            transform = ClassFactory.get_cls(ClassType.TRANSFORM, args[0])
            self.__transform__.insert(index, transform(**kwargs))
        else:
            self.__transform__.insert(index, args[0])

    def remove(self, transform_name):
        """Remove a transform from the transform_list.

        Arguments:
            transform_name (str): name of transform
        """
        if isinstance(transform_name, str):
            for trans in self.__transform__:
                if transform_name == trans.__class__.__name__:
                    self.__transform__.remove(trans)
