# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow.keras.callbacks import BroadcastGlobalVariablesCallback


class BroadcastGlobalVariablesCallbackV2(BroadcastGlobalVariablesCallback):
  """
  Keras Callback that will broadcast all global variables from root rank
  to all other processes during initialization.
  This is necessary to ensure consistent initialization of all workers when
  training is started with random weights or restored from a checkpoint.
  """

  def on_batch_end(self, batch, logs=None):
    if self.broadcast_done:
      return

    with tf.device(self.device):
      if hvd._executing_eagerly() and hasattr(self.model, 'variables'):
        # TensorFlow 2.0 or TensorFlow eager
        broadcast_vars = [var for var in self.model.variables if var.ref() not in self._local_vars]
        hvd.broadcast_variables(broadcast_vars,
                                root_rank=self.root_rank)
        # In the DeePray, some model don't contain optimizer
        # hvd.broadcast_variables(self.model.optimizer.variables(),
        #                         root_rank=self.root_rank)
      else:
        bcast_op = hvd.broadcast_global_variables(self.root_rank)
        self.backend.get_session().run(bcast_op)

    self.broadcast_done = True
