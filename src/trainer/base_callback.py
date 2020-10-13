"""
@author: Guanghan Ning
@file: base_callback.py
@time: 10/9/20 2:43 下午
@file_desc: Callbacks called at certain points of a trainer.
"""

from src.core.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.CALLBACK)
class Callback(object):
    """Abstract class for building new callbacks."""

    def __init__(self):
        """Init callback object."""
        self.trainer = None
        self.params = None

    def set_trainer(self, trainer):
        """Set trainer object for current callback."""
        self.trainer = trainer

    def set_params(self, params):
        """Set parameters for current callback."""
        self.params = params

    def before_train(self, logs=None):
        """Be called before the training process.

        Subclasses should override this for their own purposes
        """

    def before_epoch(self, epoch, logs=None):
        """Be called before each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def before_train_step(self, batch_index, logs=None):
        """Be called before each batch forward step.

        Subclasses should override this for their own purposes
        """

    def make_batch(self, batch):
        """Be called on the generation of each batch.

        Subclasses should override this for their own purposes
        This will replace the default make_batch function in the
        trainer.
        """

    def train_step(self, batch):
        """Be called on each batch training.

        Subclasses should override this for their own purposes
        This will replace the default train_step function in the
        trainer.
        """

    def valid_step(self, batch):
        """Be called on each batch validating.

        Subclasses should override this for their own purposes
        This will replace the default valid_step function in the
        valider.
        """

    def after_train_step(self, batch_index, logs=None):
        """Be called after each batch training.

        Subclasses should override this for their own purposes
        """

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch during the training process.

        Subclasses should override this for their own purposes
        """

    def after_train(self, logs=None):
        """Be called after the training process.

        Subclasses should override this for their own purposes
        """

    def before_valid(self, logs=None):
        """Be called before the validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        """

    def before_valid_step(self, batch_index, logs=None):
        """Be called before a batch evaluation or validation.

        Subclasses should override this for their own purposes

        Also called before a validation batch during the train function
        if validition is requied
        """

    def after_valid_step(self, batch_index, logs=None):
        """Be called after a batch validation.

        Subclasses should override this for their own purposes

        Also called after a validation batch during the train function,
        if validition is requied
        """

    def after_valid(self, logs=None):
        """Be called after the validation.

        Subclasses should override this for their own purposes
        """
