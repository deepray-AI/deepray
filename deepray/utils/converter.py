# -*- coding:utf-8 -*-
# Copyright 2019 The DeepRec Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import multiprocessing
import sys
import time
from datetime import datetime
from multiprocessing import Process, Queue

import pandas as pd
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSV2TFRecord(object):
    def __init__(self, LABEL, NUMERICAL_FEATURES, CATEGORY_FEATURES, VARIABLE_FEATURES, gzip=False):
        self.LABEL = LABEL
        self.NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.CATEGORY_FEATURES = CATEGORY_FEATURES
        self.VARIABLE_FEATURES = VARIABLE_FEATURES
        self.gzip = gzip
        self.in_queue = Queue()
        self.out_queue = Queue()

    def __call__(self, dataframe, out_file, prebatch=1, *args, **kwargs):
        """
        Transforms tablue data in pandas.DataFrame format to tf.Example protos and dump to a TFRecord file.
            The benefit of doing this is to use existing training and evaluating functionality within tf
            packages.

        :param dataframe:   intput pd.DataFrame data
        :param out_file:    output TFRercord file path
        :param prebatch:    batch size of examples to package into TFRecord file
        :param args:
        :param kwargs:
        :return:
        """

        def parsing_loop():
            """
            function to be executed within each parsing process.

            Args:
              in_queue: the queue used to store avazu data records as strings.
              out_queue: the queue used to store serialized tf.Examples as strings.
            """
            while True:  # loop.
                raw_record = self.in_queue.get()  # read from in_queue.
                # logging.debug('parsing_loop raw_example:{}'.format(raw_record))
                if isinstance(raw_record, str):
                    # We were done here.
                    break
                features = {}  # dict for all feature columns and target column.
                for item in raw_record.columns:
                    tmp = list(raw_record[item].values)
                    if item in self.CATEGORY_FEATURES:
                        features[item] = self._int64_feature(tmp)
                    elif item in self.VARIABLE_FEATURES:
                        features[item] = self._int64_feature(tmp[0])
                    elif item in self.NUMERICAL_FEATURES:
                        features[item] = self._float_feature(tmp)
                    elif item in self.LABEL:
                        features[item] = self._int64_feature(tmp)

                # create an instance of tf.Example.
                example = tf.train.Example(features=tf.train.Features(feature=features))
                # serialize the tf.Example to string.
                raw_example = example.SerializeToString()

                # write the serialized tf.Example out.
                self.out_queue.put(raw_example)

        def writing_loop():
            """
            function to be executed within the single writing process.

            Args:
              out_queue: the queue used to store serialized tf.Examples as strings.
              out_file: string, path to the TFRecord file for transformed tf.Example protos.
            """
            options = tf.io.TFRecordOptions(compression_type='GZIP')
            writer = tf.io.TFRecordWriter(out_file, options=options if self.gzip else None)  # writer for the output TFRecord file.
            sample_count = 0
            while True:
                raw_example = self.out_queue.get()  # read from out_queue.
                logging.debug('writing_loop raw_example:{}'.format(raw_example))
                if isinstance(raw_example, str):
                    break
                writer.write(raw_example)  # write it out.
                sample_count += 1
                if not sample_count % 1000:
                    logging.info('%s Processed %d examples' % (datetime.now(), sample_count * prebatch))
                    sys.stdout.flush()
            writer.close()  # close the writer.
            logging.info('%s >>>> Processed %d examples <<<<' % (datetime.now(), sample_count * prebatch))
            self.sample_cnt = sample_count
            sys.stdout.flush()

        start_time = time.time()
        # start parsing processes.
        num_parsers = int(multiprocessing.cpu_count() - 1)
        parsers = []
        for i in range(num_parsers):
            p = Process(target=parsing_loop)
            parsers.append(p)
            p.start()

        # start writing process.
        writer = Process(target=writing_loop)
        writer.start()
        # logging.info('%s >>>> BEGIN to feed input file %s <<<<' % (datetime.now(), self.path))

        for i in range(0, len(dataframe), prebatch):
            line = dataframe[i:i + prebatch]
            if len(line) < prebatch:
                continue
            self.in_queue.put(line)  # write to in_queue.
        # terminate and wait for all parsing processes.
        for i in range(num_parsers):
            self.in_queue.put("DONE")
        for i in range(num_parsers):
            parsers[i].join()

        # terminate and wait for the writing process.
        self.out_queue.put("DONE")
        writer.join()
        end_time = time.time()
        total_time = (end_time - start_time)
        logging.warning('Total time %.2f s, speed %.2f sample/s,'
                        ' total samples %d.' %
                        (total_time, len(dataframe) / total_time, len(dataframe)))
        logging.info('%s >>>> END of consuming input file %s <<<<' % (datetime.now(), out_file))
        sys.stdout.flush()

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def write_feature_map(self, dateframe, path):
        with open(path, 'a') as f:
            for item in self.CATEGORY_FEATURES:
                f.writelines(','.join([str(dateframe[item].max()), item, 'CATEGORICAL\n']))
            for item in self.NUMERICAL_FEATURES:
                f.write(','.join(['1', item, 'NUMERICAL\n']))
            for item in self.VARIABLE_FEATURES:
                pass
                # f.write(','.join(['1', item, 'VARIABLE\n']))
            for item in self.LABEL:
                f.write(','.join([str(dateframe[item].nunique()), item, 'LABEL\n']))
