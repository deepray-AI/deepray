#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#

"""
build and train model
"""

import sys
import time

from absl import app, flags
import deepray as dp

from deepray.base.trainer import train
from deepray.model.build_model import BuildModel

FLAGS = flags.FLAGS


def main(unused=None, flags=None):
    test = dp.model.list_available()
    print(test)
    if flags:
        FLAGS(flags, known_only=True)
    flags = FLAGS
    model = BuildModel(flags)
    history = train(model)
    print(history)
    # model.predict()


def runner(argv=None):
    if len(argv) <= 1:
        argv = [
            sys.argv[0],
            '--model=fm',
            '--optimizer=lazyadam',
            '--train_data=/Users/vincent/Projects/DeePray_Keras/examples/census/data/train',
            '--valid_data=/Users/vincent/Projects/DeePray_Keras/examples/census/data/valid',
            '--feature_map=/Users/vincent/Projects/DeePray_Keras/examples/census/data/feature_map.csv',
            '--learning_rate=0.01',
            '--epochs=1',
            '--steps_per_summary=1',
            '--gzip=False',
            '--patient_valid_passes=3',
            '--prebatch=1',
            '--parallel_reads_per_file=1',
            '--parallel_parse=1',
            '--interleave_cycle=1',
            '--prefetch_buffer=16',
            '--batch_size=64',
            '--deep_layers=100,50',
            '--model_path=./outputs',
            '--summaries_dir=/Users/vincent/Projects/DeePray_Keras/examples/census/summaries/{}'.format(
                time.strftime('%y%m%d%H%M')),
            '--alsologtostderr=True'
        ]
    main(flags=argv)


if __name__ == "__main__":
    app.run(runner)
