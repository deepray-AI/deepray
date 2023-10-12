import tensorflow as tf

print(tf.__version__)


def frozen_keras_graph(model):
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
  from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

  real_model = tf.function(model).get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

  input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != tf.resource]
  output_tensors = frozen_func.outputs

  graph_def = run_graph_optimizations(
      graph_def,
      input_tensors,
      output_tensors,
      config=get_grappler_config(["constfold", "function"]),
      graph=frozen_func.graph
  )

  return graph_def


keras = tf.keras


class MyCustomLayer(keras.layers.Layer):

  def __init__(self):
    super(MyCustomLayer, self).__init__(self)
    self._weight = tf.Variable(initial_value=(2., 3.))

  def call(self, input):
    output = tf.sigmoid(input) * self._weight
    return output


model = keras.models.Sequential([keras.layers.Input((1, 2)), MyCustomLayer()])

graph_def = frozen_keras_graph(model)

# frozen_func.graph.as_graph_def()
tf.io.write_graph(graph_def, '.', 'frozen_graph.pb')
