"""
@author: Guanghan Ning
@file: generator.py
@time: 10/14/20 10:23 下午
@file_desc: Generator for NasPipeStep."""

import glog as logging
from datetime import datetime

from src.search_algorithms.base_algorithm import SearchAlgorithm
from src.search_space.search_space import SearchSpace
#from vega.core.report import Report, ReportRecord

from src.utils.read_configure import Config, dict2config
from src.utils.utils_dict import update_dict


class TaskConfig(dict):
    """Task Config."""

    task_id = datetime.now().strftime('%m%d.%H%M%S.%f')[:-3]
    local_base_path = "./tasks"
    output_subpath = "output"
    best_model_subpath = "best_model"
    log_subpath = "logs"
    result_subpath = "result"
    worker_subpath = "workers/[step_name]"
    backup_base_path = None
    use_dloop = False


class General(object):
    """General Config."""

    task = TaskConfig
    step_name = None
    backend = 'pytorch'
    device_category = 'GPU'
    env = None
    calc_params_each_epoch = False


class Generator(object):
    """Convert search space and search algorithm, sample a new model from the search space as NAS trainer."""

    def __init__(self):
        self.step_name = General.step_name
        self.search_space = SearchSpace()
        self.search_alg = SearchAlgorithm(self.search_space.search_space)

        # TODO: record search results
        '''
        self.report = Report()
        self.record = ReportRecord()
        self.record.step_name = self.step_name
        if hasattr(self.search_alg.config, 'objective_keys'):
            self.record.objective_keys = self.search_alg.config.objective_keys
        '''

    @property
    def is_completed(self):
        """Define a property to determine search algorithm is completed."""
        return self.search_alg.is_completed

    def sample(self):
        """Sample a work id and model from search algorithm."""
        res = self.search_alg.search()
        if not res:
            return None
        if not isinstance(res, list):
            res = [res]
        out = []
        for sample in res:
            if isinstance(sample, tuple):
                sample = dict(worker_id=sample[0], desc=sample[1])
                import glog as log
                log.info("sample['desc']: {}".format(sample['desc']))
        #    record = self.record.load_dict(sample)
        #    logging.debug("Broadcast Record=%s", str(record))
        #    Report().broadcast(record)
        #    desc = self._decode_hps(record.desc)
        #    out.append((record.worker_id, desc))
        #return out
            #desc = self._decode_hps(sample['desc'])
            desc = sample['desc']
            log.info("desc: {}".format(desc))
            out.append((0, desc))
        return out

    '''
    def update(self, step_name, worker_id):
        """Update search algorithm accord to the worker path.

        Arguments:
            step_name: step name
            worker_id: current worker id
        """
        report = Report()
        record = report.receive(step_name, worker_id)
        logging.debug("Get Record=%s", str(record))
        self.search_alg.update(record.serialize())
        report.dump_report(record.step_name, record)
        logging.info("Update Success. step_name=%s, worker_id=%s", step_name, worker_id)
        logging.info("Best values: %s", Report().pareto_front(step_name=General.step_name))
    '''

    @staticmethod
    def _decode_hps(hps):
        """Decode hps: `trainer.optim.lr : 0.1` to dict format.

        This Config will be override in Trainer or Datasets class
        The override priority is: input hps > user configuration >  default configuration

        Arguments:
            hps: hyper params

        Return:
            dict
        """
        hps_dict = {}
        if hps is None:
            return None
        if isinstance(hps, tuple):
            return hps
        for hp_name, value in hps.items():
            hp_dict = {}
            for key in list(reversed(hp_name.split('.'))):
                if hp_dict:
                    hp_dict = {key: hp_dict}
                else:
                    hp_dict = {key: value}
            # update cfg with hps
            #hps_dict = update_dict(hps_dict, hp_dict, [])
            hps_dict = update_dict(hp_dict, hps_dict, [])

        hps_config = Config()
        hps_config = dict2config(config_dst=hps_config, dict_src=hps_dict)
        return hps_config
