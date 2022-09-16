"""
@author: Guanghan Ning
@file: utils_saver.py
@time: 10/18/20 7:59 下午
@file_desc:
"""
# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import json
import logging
import os
import glob
import traceback
from copy import deepcopy
import numpy as np
import pandas as pd
import pareto

from functools import wraps
from collections import OrderedDict
from collections import defaultdict

from src.core.base_task import Task
from src.core.default_config import GeneralConfig
from src.utils.utils_nsga import SortAndSelectPopulation
from src.utils.utils_record import Record
from src.utils.utils_io_folder import create_folder, copy_file


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance


@singleton
class Saver(object):
    """Report class to save all records and broadcast records to share memory."""

    _hist_records = OrderedDict()
    REPORT_FILE_NAME = 'reports'
    BEST_FILE_NAME = 'best'
    exist_dict = defaultdict()

    def add(self, record):
        """Add one record into set."""
        self._hist_records[record.uid] = record

    @property
    def all_records(self):
        """Get all records."""
        return deepcopy(list(self._hist_records.values()))

    def pareto_front(self, step_name=None, nums=None, records=None):
        """Get parent front. pareto."""
        if records is None:
            records = self.all_records
            records = list(filter(lambda x: x.step_name == step_name and x.performance is not None, records))
        in_pareto = [record.rewards if isinstance(record.rewards, list) else [record.rewards] for record in records]
        if not in_pareto:
            return None, None
        try:
            fitness = np.array(in_pareto)
            if fitness.shape[1] != 1 and nums is not None and len(in_pareto) > nums:
                # len must larger than nums, otherwise dead loop
                _, res, selected = SortAndSelectPopulation(fitness.T, nums)
            else:
                outs = pareto.eps_sort(fitness, maximize_all=True, attribution=True)
                res, selected = np.array(outs)[:, :-2], np.array(outs)[:, -1].astype(np.int32)
            return res.tolist(), selected.tolist()
        except Exception as ex:
            logging.error('No pareto_front_records found, ex=%s', ex)
            return [], []

    def get_step_records(self, step_name=None):
        """Get step records."""
        if not step_name:
            step_name = GeneralConfig.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps, records))
        return records

    def get_pareto_front_records(self, step_name=None, nums=None):
        """Get Pareto Front Records."""
        if not step_name:
            step_name = GeneralConfig.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps and x.performance is not None, records))
        outs, selected = self.pareto_front(step_name, nums, records=records)
        if not outs:
            return []
        else:
            return [records[idx] for idx in selected]

    def dump_report(self, step_name=None, record=None):
        """Save one records."""
        try:
            if record and step_name:
                self._append_record_to_csv(self.REPORT_FILE_NAME, step_name, record.serialize())
            self.backup_output_path()
        except Exception:
            logging.warning(traceback.format_exc())

    def output_pareto_front(self, step_name, desc=True, weights_file=False, performance=False):
        """Save one records."""
        logging.debug("All records in report, records={}".format(self.all_records))
        records = deepcopy(self.get_pareto_front_records(step_name))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump pareto front records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)

    def output_step_all_records(self, step_name, desc=True, weights_file=False, performance=False):
        """Output step all records."""
        records = self.all_records
        logging.debug("All records in report, records={}".format(self.all_records))
        records = list(filter(lambda x: x.step_name == step_name, records))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump records, report is emplty.")
            return
        self._output_records(step_name, records, desc, weights_file, performance)

    def _output_records(self, step_name, records, desc=True, weights_file=False, performance=False):
        """Dump records."""
        columns = ["worker_id", "performance", "desc"]
        outputs = []
        for record in records:
            record = record.serialize()
            _record = {}
            for key in columns:
                _record[key] = record[key]
            outputs.append(deepcopy(_record))
        data = pd.DataFrame(outputs)
        step_path = os.path.join("output", step_name)
        create_folder(step_path)
        _file = os.path.join(step_path, "output.csv")

        try:
            data.to_csv(_file, index=False)
        except Exception:
            logging.error("Failed to save output file, file={}".format(_file))
        for record in outputs:
            worker_id = record["worker_id"]
            worker_path = os.path.join(step_name, worker_id)
            outputs_globs = []
            if desc:
                outputs_globs += glob.glob(os.path.join(worker_path, "desc_*.json"))
            if weights_file:
                outputs_globs += glob.glob(os.path.join(worker_path, "model_*.pth"))
            if performance:
                outputs_globs += glob.glob(os.path.join(worker_path, "performance_*.json"))
            for _file in outputs_globs:
                copy_file(_file, step_path)

    @classmethod
    def _save(cls, record):
        """save one record."""
        if not record:
            logging.warning("Record is None.")
            return
        cls().add(record)
        cls._save_worker_record(record.serialize())

    @classmethod
    def receive(cls, step_name, worker_id):
        """Get value."""
        key = "{}.{}".format(step_name, worker_id)
        if key not in cls.exist_dict:
            record = Record(step_name, worker_id)
            cls.exist_dict[key] = record
        else:
            record = cls.exist_dict[key]
        cls().add(record)
        return record

    @classmethod
    def _save_worker_record(cls, record):
        step_name = record.get('step_name')
        worker_id = record.get('worker_id')
        _path = os.path.join("output", str(step_name), str(worker_id))
        for record_name in ["desc", "performance"]:
            _file_name = None
            _file = None
            record_value = record.get(record_name)
            if not record_value:
                continue
            _file = None
            try:
                # for cars/darts save multi-desc
                if isinstance(record_value, list) and record_name == "desc":
                    for idx, value in enumerate(record_value):
                        _file_name = "desc_{}.json".format(idx)
                        _file = os.path.join(_path, _file_name)
                        with open(_file, "w") as f:
                            json.dump(record_value, f)
                else:
                    _file_name = None
                    if record_name == "desc":
                        _file_name = "desc_{}.json".format(worker_id)
                    if record_name == "performance":
                        _file_name = "performance_{}.json".format(worker_id)
                    _file = os.path.join(_path, _file_name)
                    with open(_file, "w") as f:
                        json.dump(record_value, f)
            except Exception as ex:
                logging.error("Failed to save {}, file={}, desc={}, msg={}".format(
                    record_name, _file, record_value, str(ex)))

    def __repr__(self):
        """Override repr function."""
        return str(self.all_records)


    def csv_to_records(self, csv_file_path, step_name=None, record_name='best'):
        """Transfer cvs_file to records."""
        local_output_path = ''
        if not csv_file_path and not step_name:
            return []
        elif csv_file_path:
            local_output_path = csv_file_path
        if (not os.path.exists(local_output_path) or local_output_path) and step_name:
            local_output_path = os.path.join("output", step_name)
        csv_file_path = os.path.join(local_output_path, "{}.csv".format(record_name))
        logging.info("csv_file_path: {}".format(csv_file_path))
        if not os.path.isfile(csv_file_path):
            return []
        csv_headr = pd.read_csv(csv_file_path).columns.values
        csv_value = pd.read_csv(csv_file_path).values
        records = []
        for item in csv_value:
            record = dict(zip(csv_headr, item))
            records.append(Record().load_dict(record))
        logging.info("csv_to_records: {}".format(records))
        return records

    @classmethod
    def load_records_from_model_folder(cls, model_folder):
        """Transfer json_file to records."""
        if not model_folder or not os.path.exists(model_folder):
            logging.error("Failed to load records from model folder, folder={}".format(model_folder))
            return []
        records = []
        pattern = os.path.join(model_folder, "desc_*.json")
        files = glob.glob(pattern)
        for _file in files:
            try:
                with open(_file) as f:
                    worker_id = _file.split(".")[-2].split("_")[-1]
                    #weights_file = os.path.join(os.path.dirname(_file), "model_{}.pth".format(worker_id))
                    weights_file = os.path.join("output", "checkpoint.pth")
                    if os.path.exists(weights_file):
                        sample = dict(worker_id=worker_id, desc=json.load(f), weights_file=weights_file)
                    else:
                        sample = dict(worker_id=worker_id, desc=json.load(f))
                    record = Record().load_dict(sample)
                    records.append(record)
            except Exception as ex:
                logging.info('Can not read records from json because {}'.format(ex))
        return records

