"""
@author: Guanghan Ning
@file: base_trainer.py
@time: 10/9/20 2:43 下午
@file_desc:
"""
import os
import pickle
import glog as log
import torch

from src.core.class_factory import ClassType, ClassFactory

from src.utils.utils_log import init_log
from src.utils.read_configure import Config, class2config, desc2config
from src.utils.utils_io_folder import create_folder, copy_folder

from src.trainer.base_worker import Worker
from src.trainer.base_callback_list import CallbackList
from src.trainer.optimizer import Optimizer
from src.trainer.lr_scheduler import LrScheduler
from src.trainer.loss import Loss
from src.trainer.base_metrics import Metrics

from src.search_space.description import NetworkDesc
from src.core.default_config import OptimConfig, LrSchedulerConfig, MetricsConfig, LossConfig, TrainerConfig


class DefaultTrainerConfig(object):
    """Default Trainer Config."""
    # GPU
    cuda = True
    device = cuda if cuda is not True else 0
    # Model
    pretrained_model_file = None
    save_model_desc = False
    # Report
    report_freq = 10
    # Training
    seed = 0
    epochs = 1
    optim = OptimConfig()
    lr_scheduler = LrSchedulerConfig()
    metric = MetricsConfig()
    loss = LossConfig()
    # Validation
    with_valid = True
    valid_interval = 1
    # evaluation
    perfs_cmp_mode = None
    perfs_cmp_key = None
    call_metrics_on_train = True

    grad_clip = None
    #model_statistics = True
    model_statistics = False
    callbacks = None
    #callbacks = [DartsTrainer]
    darts_template_file = "src/baselines/baseline_darts.json"
    lr_adjustment_position = 'after_epoch'


@ClassFactory.register(ClassType.TRAINER)
class Trainer(Worker):
    config = TrainerConfig()
    #config = DefaultTrainerConfig()

    def __init__(self, model=None, hps=None, load_ckpt_flag=False, **kwargs):
        super(Trainer, self).__init__()
        # Data Member list of Trainer
        self.use_cuda = self.config.cuda
        self.do_validation = True
        self.auto_save_ckpt = True
        self.auto_save_perf = True
        self.skip_train = False

        self.hps = hps
        self.optimizer = None
        self.lr_scheduler = None

        self.model = model
        self.loss = None

        self.train_loader = None
        self.valid_loader = None
        self.train_step = None
        self.valid_step = None
        self.train_metrics = None
        self.valid_metrics = None

        self.epochs = self.config.epochs
        self.with_valid = self.config.with_valid
        self.valid_interval = self.config.valid_interval

        self.make_batch = None
        self.callbacks = None
        self.performance = None

        self.model_desc = {}
        self.visual_data = {}
        self.load_ckpt_flag = load_ckpt_flag
        self.checkpoint_file_name = 'checkpoint.pth'
        self.model_pickle_file_name = 'model.pkl'
        self.model_path = os.path.join(self.model_pickle_file_name)
        self.checkpoint_file = os.path.join(self.checkpoint_file_name)
        self.weights_file = os.path.join("model.pth")

        # Indicate whether the necessary components of a trainer
        # has been built for running
        self.is_built = False
        self.config.kwargs = kwargs

    def train_process(self):
        """Whole train process of the TrainWorker specified in config.

        After training, the model and validation results are saved to local_worker_path and s3_path.
        """
        log = init_log("info", log_file="worker.txt")

        log.info("Start training process, building components (optimizer, lr_scheduler, etc.)")
        self.build(model=self.model, hps=self.hps)

        log.info("self.config.callbacks: {}".format(self.config.callbacks))
        self._init_callbacks(self.callbacks)
        log.info("Trainer's callbacks: {}".format(self.callbacks))

        self._train_loop()

    def build(self, model=None, optimizer=None, loss=None,
              lr_scheduler=None, metrics=None, hps=None,
              callbacks=None, train_loader=None, valid_loader=None,
              make_batch=None, train_step=None, valid_step=None,
              checkpoint_file_name="checkpoint.pth",
              model_pickle_file_name="model.pkl"):
        """Build the trainer by assembling the necessary components."""
        # Initialize hyper-parameters by parameters or configurations
        log.info("Init Trainer config with HPS config: {}".format(hps))
        self.config = self._init_hps(hps)

        trainer_config = Config()
        trainer_config = class2config(config_dst=trainer_config, class_src=self.config)
        log.info("Trainer config: {}".format(trainer_config))

        self.checkpoint_file_name = checkpoint_file_name
        self.model_pickle_file_name = model_pickle_file_name

        self._init_step_functions(make_batch, train_step, valid_step)
        self._init_cuda_setting()

        self.model = self._init_model(model)

        if self.load_ckpt_flag:
            self._load_checkpoint()
        else:
            self._load_pretrained_model()

        self.train_loader = self._init_dataloader(mode='train', loader=train_loader)
        self.valid_loader = self._init_dataloader(mode='val', loader=valid_loader)

        self.optimizer = Optimizer()(model=self.model) if optimizer is None else optimizer
        self.loss = Loss()() if loss is None else loss
        self.lr_scheduler = LrScheduler()(self.optimizer) if lr_scheduler is None else lr_scheduler

        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics(metrics)
        self.valid_metrics = self._init_metrics(metrics)

        if self.callbacks is None:
            self.callbacks = callbacks

        if self.step_name:
            cur_working_dir = os.path.join("output", self.step_name)
        else:
            cur_working_dir = os.path.join("output", "default_nas")

        create_folder(cur_working_dir)
        # Make sure Trainer has been built for training
        self.is_built = True

    def _init_step_functions(self, make_batch=None,
                             train_step=None, valid_step=None):
        # Init make_batch function by user or using the default one
        if self.make_batch is None:
            if make_batch is not None:
                self.make_batch = make_batch
            else:
                self.make_batch = self._default_make_batch

        # Init train_step function by user or using the default one
        if self.train_step is None:
            if train_step is not None:
                self.train_step = train_step
            else:
                self.train_step = self._default_train_step

        # Init valid_step function by user or using the default one
        if self.valid_step is None:
            if valid_step is not None:
                self.valid_step = valid_step
            else:
                self.valid_step = self._default_valid_step

    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        input, target = batch
        if self.use_cuda:
            input, target = input.cuda(), target.cuda()
        return input, target

    def _default_train_step(self, batch):
        input, target = batch
        self.optimizer.zero_grad()

        output = self.model(input)
        loss = self.loss(output, target)
        loss.backward()

        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.config.grad_clip)
        self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output}

    def _default_valid_step(self, batch):
        input, target = batch
        output = self.model(input)
        return {'valid_batch_output': output}

    def _init_cuda_setting(self):
        """Init CUDA setting."""
        if not self.config.cuda:
            self.config.device = -1
            return
        self.config.device = self.config.cuda if self.config.cuda is not True else 0
        self.use_cuda = True
        torch.cuda.manual_seed(self.config.seed)

    def _init_hps(self, hps=None):
        """Load hyper-parameter search config from file."""
        if hps is not None:
            self.hps = hps
        elif self.config.hps_file is not None:
            desc_file = self.config.hps_file.replace("{local_base_path}", self.local_base_path)
            self.hps = Config(desc_file)

        if self.hps and self.hps.get('trainer'):
            self.config = desc2config(config_dst=self.config, desc_src=self.hps.get('trainer'))
        return self.config

    def _init_model(self, model=None):
        """Load model desc from save path and parse to model."""
        # init with model if it is given
        if model is not None:
            if self.use_cuda:
                model = model.cuda()
            return model

        # get model description based on config
        model_cfg = Config(ClassFactory.__configs__.get('model'))
        if "model_desc" in model_cfg and model_cfg.model_desc is not None:
            model_desc = model_cfg.model_desc
        else:
            return None

        # get pytorch model based on model description
        if model_desc is not None:
            self.model_desc = model_desc
            net_desc = NetworkDesc(model_desc)
            model = net_desc.to_model()
            if self.use_cuda:
                model = model.cuda()
            return model
        else:
            return None

    def _load_pretrained_model(self):
        """ Load pretrained model based on ckpt path in config"""
        if self.model is None:
            log.info("Model is None, not pretrained model loaded.")
            return

        if self.config.pretrained_model_file is not None:
            model_file = self.config.pretrained_model_file
            model_file = os.path.abspath(model_file)

            ckpt = torch.load(model_file)
            self.model.load_state_dict(ckpt)
            return

    def _load_checkpoint(self, saved_folder=None):
        """Load checkpoint."""
        if saved_folder is None:
            log.error("Save folder for checkpoint is not given.")
            return

        checkpoint_file = os.path.join(saved_folder, self.checkpoint_file_name)
        model_pickle_file = os.path.join(saved_folder, self.model_pickle_file_name)

        try:
            with open(model_pickle_file, 'rb') as f:
                model = pickle.load(f)
                ckpt = torch.load(checkpoint_file, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['weight'])
                if self.config.cuda:
                    model = model.cuda()
                self.model = model
        except Exception:
            log.info('Checkpoint file does not exist or is broken; using default model now.')
            return

    def _init_dataloader(self, mode, loader=None):
        """Init dataloader."""
        if loader is not None:
            return loader

        if mode == "train" and self.hps is not None and self.hps.get("dataset") is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode, hp=self.hps.get("dataset"))
        else:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            dataset = dataset_cls(mode=mode)

        return dataset.dataloader

    def _init_metrics(self, metrics=None):
        """Init metrics."""
        if metrics is not None:
            return metrics
        else:
            return Metrics()

    def _init_callbacks(self, callbacks):
        # Initialize custom callbacks by configuration or parameters
        if callbacks is not None:
            return callbacks
        disables = []
        if not self.config.model_statistics:
            disables.append('ModelStatistics')
        self.callbacks = CallbackList(self.config.callbacks, disables)
        self.callbacks.set_trainer(self)

    def _backup(self):
        """Backup result worker folder."""
        if self.need_backup is True and self.backup_base_path is not None:
            backup_worker_path = os.path.join(self.backup_base_path,
                                              self.get_worker_subpath())
            copy_folder(src= self.get_local_worker_path(self.step_name),
                        dst= backup_worker_path)

    def _train_loop(self):
        """Perform training with data, callbacks and step functions etc."""
        # Phase 1: Pre-processing before overall training starts
        self.callbacks.before_train()

        for epoch in range(self.epochs):
            epoch_logs = {'train_num_batches': len(self.train_loader)}
            if self.with_valid:
                epoch_logs.update({'valid_num_batches': len(self.valid_loader)})
            # Phase 2: Pre-processing before one epoch
            self.callbacks.before_epoch(epoch, epoch_logs)

            # Phase 3: train for one epoch
            self._train_epoch()

            # Phase 4: validation
            if self.with_valid and self._should_run_validation(epoch):
                self._valid_epoch()

            # Phase 5: Post-processing after one epoch
            self.callbacks.after_epoch(epoch)

        # Phase 6: Post-processing as overall training ends
        self.callbacks.after_train()

    def _train_epoch(self):
        """Training logic within each epoch"""
        # Activate training mode
        self.model.train()

        # Perform training for one epoch
        for batch_index, batch in enumerate(self.train_loader):
            # Fetch a batch of data
            batch = self.make_batch(batch)
            batch_logs = {'train_batch': batch}

            # Perform logic before backprop (calculating gradients and updating parameters)
            self.callbacks.before_train_step(batch_index, batch_logs)

            # Perform backprop
            train_batch_output = self.train_step(batch)
            batch_logs.update(train_batch_output)

            # Perform logic after backprop
            self.callbacks.after_train_step(batch_index, batch_logs)

    def _valid_epoch(self):
        # Perform validation for one epoch
        self.callbacks.before_valid()
        valid_logs = None

        # Activate test mode
        self.model.eval()

        with torch.no_grad():
            for batch_index, batch in enumerate(self.valid_loader):
                # Fetch a batch of data
                batch = self.make_batch(batch)
                batch_logs = {'valid_batch': batch}

                # Perform pre-processing logic before validation on each batch
                self.callbacks.before_valid_step(batch_index, batch_logs)

                # Perform validation step on one batch
                valid_batch_output = self.valid_step(batch)

                # Perform post-processing after validation on each batch
                self.callbacks.after_valid_step(batch_index, valid_batch_output)

        # Perform post-processing after validation on a whole dataset
        self.callbacks.after_valid(valid_logs)


