"""
@author: Guanghan Ning
@file: fullytrain_pipestep.py
@time: 10/18/20 7:14 下午
@file_desc: Pipeline step for training a network with its architecture fixed.
"""

import glog as log
from src.core.base_pipestep import PipeStep
from src.core.class_factory import ClassFactory, ClassType
from src.search_space.description import NetworkDesc
from src.utils.utils_record import Record
from src.utils.utils_saver import Saver


@ClassFactory.register(ClassType.PIPE_STEP)
class FullyTrainPipeStep(PipeStep):
    """FullyTrainPipeStep is the pipeline step where the network is trained from scratch with its architecture being fixed.

    """

    def __init__(self):
        super().__init__()
        self.need_evaluate = self._has_evaluator()
        log.info("init FullyTrainPipeStep...")

    def run(self):
        """Start to run fully train with horovod or local trainer."""
        log.info("FullyTrainPipeStep started...")
        #cls_trainer = ClassFactory.get_cls('trainer')

        records = self._get_current_step_records()
        log.info("Loading pipestep records: {}".format(records))

        log.info("Training with network description: {}".format(records[-1].desc))
        self._train_model(records[-1].desc)

        log.info("Updating Record: {}".format(records))
        for record in records:
            Saver().update_report({"step_name": record.step_name, "worker_id": record.worker_id})

        log.info("Saving Record for step: {}".format(self.task.step_name))
        Saver().output_step_all_records(step_name=self.task.step_name,
                                         weights_file=True,
                                         performance=True)
        Saver().backup_output_path()

    def _get_current_step_records(self):
        step_name = self.task.step_name
        models_folder = "output/"

        # records = Saver().get_pareto_front_records(PipelineConfig.steps[cur_index - 1])
        records = Saver().load_records_from_model_folder(models_folder)

        log.info("Records: {}".format(records))
        for record in records:
            record.step_name = step_name
        return records

    def _train_model(self, model_desc=None, model_id=None):
        cls_trainer = ClassFactory.get_cls('trainer')
        if cls_trainer is None:
            cls_trainer = ClassFactory.get_cls('trainer', t_cls_name="Trainer")

        log.info(model_desc)
        if model_desc is not None:
            model = NetworkDesc(model_desc).to_model()
            log.info("Model: {}".format(model))
            trainer = cls_trainer(model, model_id)
        else:
            trainer = cls_trainer(None, 0)

        log.info("Using trainer: {}".format(type(trainer).__name__))
        trainer.train_process()

    def _evaluate_model(self, record):
        cls_evaluator = ClassFactory.get_cls('evaluator')
        evaluator = cls_evaluator({"step_name": record.step_name, "worker_id": record.worker_id})
        log.info("submit evaluator, step_name={}, worker_id={}".format(
            record.step_name, record.worker_id))
        evaluator.run()

    def _has_evaluator(self):
        try:
            ClassFactory.get_cls('evaluator')
            return True
        except Exception:
            return False

