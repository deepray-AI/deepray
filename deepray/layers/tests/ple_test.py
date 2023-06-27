import unittest
import tensorflow as tf
from ..ple import PLE


class TestPLE(unittest.TestCase):
  def setUp(self):
    self.ple = PLE(task_names=['task1', 'task2'], num_experts=2, shared_num_experts=2)

  def test_call(self):
    task_input = {
      'task1': tf.random.normal((32, 256)),
      'task2': tf.random.normal((32, 256))
    }
    task_gate_input = {
      'task1': tf.random.normal((32, 4)),
      'task2': tf.random.normal((32, 4))
    }
    shared_input = tf.random.normal((32, 256))
    shared_gate_input = tf.random.normal((32, 8))
    is_last = False

    task_output, shared_output = self.ple(task_input, task_gate_input,
                                          shared_input, shared_gate_input,
                                          is_last)
    self.assertEqual(task_output['task1'].shape, (32, 256))
    self.assertEqual(task_output['task2'].shape, (32, 256))
    self.assertEqual(shared_output.shape, (32, 256))

  def test_call_is_last(self):
    task_input = {
      'task1': tf.random.normal((32, 256)),
      'task2': tf.random.normal((32, 256))
    }
    task_gate_input = {
      'task1': tf.random.normal((32, 4)),
      'task2': tf.random.normal((32, 4))
    }
    shared_input = tf.random.normal((32, 256))
    shared_gate_input = tf.random.normal((32, 8))
    is_last = True

    task_output = self.ple(task_input, task_gate_input,
                           shared_input, shared_gate_input,
                           is_last)
    self.assertEqual(task_output['task1'].shape, (32, 256))
    self.assertEqual(task_output['task2'].shape, (32, 256))

  def test_build(self):
    self.ple.build(input_shape=(None,))
    self.assertTrue(hasattr(self.ple.gates['task1'], 'kernel'))
    self.assertTrue(hasattr(self.ple.shared_experts[0], 'kernel'))
    self.assertTrue(hasattr(self.ple.task_experts['task1'][0], 'kernel'))
    self.assertTrue(hasattr(self.ple.gate_shared, 'kernel'))


if __name__ == '__main__':
  unittest.main()
