"""
@author: Guanghan Ning
@file: nas_pipestep.py
@time: 10/14/20 8:42 下午
@file_desc: Nas Pipe Step defined in Pipeline.
"""
import time
import glog as log
from copy import deepcopy

from src.core.base_pipestep import PipeStep
from src.core.default_config import PipeStepConfig
from src.core.class_factory import ClassFactory, ClassType
from src.utils.utils_dict import update_dict
from src.search_space.description import NetworkDesc

from .generator import Generator


@ClassFactory.register(ClassType.PIPE_STEP)
class NasPipeStep(PipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        log.info("Initialize Generator.")
        self.generator = Generator()

    def run(self):
        """Do the main task in this pipe step."""
        log.info("NasPipeStep started...")
        while not self.generator.is_completed:
            samples = self.generator.sample()
            log.info("samples: {}".format(samples))
            if samples:
                for (id_ele, desc) in samples:
                    log.info("desc: {}".format(desc))

                    # model
                    if "modules" in desc:
                        PipeStepConfig.model.model_desc = deepcopy(desc)
                        log.info("PipeStep's model config: {}".format(PipeStepConfig.model.__dict__))
                    elif "network" in desc:
                        origin_desc = PipeStepConfig.model.model_desc
                        desc = update_dict(desc["network"], origin_desc)
                        log.info("PipeStep's network config: {}".format(desc["Network"]))

                    model = NetworkDesc(desc).to_model()
                    log.info("Model: {}".format(model))

                    # trainer
                    cls_trainer = ClassFactory.get_cls('trainer')
                    trainer = cls_trainer(model, hps=desc)
                    log.info("Using trainer: {}".format(type(trainer).__name__))
                    trainer.train_process()
            else:
                time.sleep(0.5)

        # TODO
        '''
        cls_evaluator = ClassFactory.get_cls('evaluator')
        evaluator = cls_evaluator()
        evaluator.evaluate_process()
        self.update_generator(self.generator)

        logging.info("Pareto_front values: %s", Report().pareto_front(General.step_name))
        Report().output_pareto_front(General.step_name)
        '''

    # TODO
    ''' 
    # evaluate each sample run in order to compare and select from paleto front
    def evaluate(self, wait_until_finish):
        try:
            cls_evaluator = ClassFactory.get_cls('evaluator')
        except Exception as e:
            cls_evaluator = None
            logging.warning("Get evaluator failed:{}".format(str(e)))

        if cls_evaluator is not None:
            evaluator = cls_evaluator()
            evaluator.evaluate_process() #TODO
        self.update_generator(self.generator)


    def update_generator(self, generator, worker_info):
        """Get finished worker's info, and use it to update target `generator`.

        Will get the worker's working dir, and then call the function
        `generator.update(step_name)`.

        Arguments:
            Generator generator: The target `generator` need to update.
            worker_info: `worker_info` is the finished worker's info, usually
            a dict or list of dict include `step_name`

        Return:
            worker_info: dict or list of dict.

        """
        if worker_info is None:
            return
        if not isinstance(worker_info, list):
            worker_info = [worker_info]
        for one_info in worker_info:
            step_name = one_info["step_name"]
            logging.info("update generator, step name: {}".format(step_name))
            try:
                generator.update(step_name, worker_id = -1)
            except Exception:
                logging.error("Failed to upgrade generator, step_name={}, worker_id={}.".format(step_name))
                logging.error(traceback.format_exc())
    '''
