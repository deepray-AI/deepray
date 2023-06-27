from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras import layers


class PLE(tf.keras.layers.Layer):
  def __init__(self, task_names: List[str], num_experts: int, shared_num_experts: int):
    super(PLE, self).__init__()
    self.task_names = task_names
    self.task_num_experts = num_experts
    self.shared_num_experts = shared_num_experts

    self.gates = {task: layers.Dense(self.task_num_experts + self.shared_num_experts,
                                     activation='softmax', name=f'gate_{task}')
                  for task in self.task_names}
    self.shared_experts = [layers.Dense(256, activation='relu', name=f'shared_expert_{i}')
                           for i in range(self.shared_num_experts)]
    self.task_experts = {task: [layers.Dense(256, activation='relu', name=f'{task}_expert_{i}')
                                for i in range(self.task_num_experts)]
                         for task in self.task_names}
    self.gate_shared = layers.Dense(self.task_num_experts * len(self.task_names) + self.shared_num_experts,
                                    activation='softmax', name='gate_shared')

  def call(self,
           task_input: Dict[str, tf.Tensor],
           task_gate_input: Dict[str, tf.Tensor],
           shared_input: tf.Tensor,
           shared_gate_input: tf.Tensor,
           is_last: bool) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    # Calculate gate coefficients
    gates = {task: self.gates[task](task_gate_input[task]) for task in self.task_names}

    # Calculate shared expert outputs
    shared_experts = [tf.expand_dims(self.shared_experts[i](shared_input), axis=2)
                      for i in range(self.shared_num_experts)]
    task_experts = {task: [tf.expand_dims(self.task_experts[task][i](concats), axis=2)
                           for i in range(self.task_num_experts)]
                    for task, concats in task_input.items()}
    # Calculate task expert outputs
    task_expert_out = {
      task: tf.concat(task_experts[task] + shared_experts, axis=2)
      for task in self.task_names
    }  # (batch, 256, NUM_EXPERTS)
    if not is_last:
      # Calculate shared gate coefficients
      gate_shared = self.gate_shared(shared_gate_input)
      # Concatenate all expert outputs
      shared_expert_out = tf.concat(
        sum([task_experts[task] for task in self.task_names], []) + shared_experts, axis=2)
      # Calculate final output
      return {task: tf.keras.backend.batch_dot(task_expert_out[task], gates[task], axes=(2, 1))
              for task in self.task_names}, tf.keras.backend.batch_dot(shared_expert_out, gate_shared,
                                                                       axes=(2, 1))
    else:
      # Calculate final output
      return {task: tf.keras.backend.batch_dot(task_expert_out[task], gates[task], axes=(2, 1))
              for task in self.task_names}
