# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import datetime
import os
logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(log_file,exist_ok=True)
        log_file =  os.path.join(log_file,'log-{}.txt'.format(run_start_time))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
