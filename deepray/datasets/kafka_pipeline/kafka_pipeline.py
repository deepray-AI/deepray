import multiprocessing
import sys
from abc import ABC

import tensorflow as tf
from tensorflow_io.python.ops import core_ops

from deepray.datasets.datapipeline import DataPipeline
from deepray.utils import logging_util

logger = logging_util.get_logger()


class KafkaGroupIODataset(tf.data.Dataset):
  """Represents a streaming dataset from kafka using consumer groups.

  The dataset is created by fetching messages from kafka using consumer clients
  which are part of a consumer group. Owing to the offset management capability of
  the kafka brokers, the dataset can maintain offsets of all the partitions
  without explicit initialization. If the consumer client joins an existing
  consumer group, it will start fetching messages from the already committed offsets.
  To start fetching the messages from the beginning, please join a different consumer group.
  The dataset will be prepared from the committed/start offset until the last offset.

  The dataset can be prepared and iterated in the following manner:

  >>> import tensorflow_io as tfio
  >>> dataset = tfio.experimental.streaming.KafkaGroupIODataset(
                      topics=["topic1"],
                      group_id="cg",
                      servers="localhost:9092"
                  )

  >>> for (message, key) in dataset:
  ...     print(message)

  Cases may arise where the consumer read time out issues arise due to
  the consumer group being in a rebalancing state. In order to address that, please
  set `session.timeout.ms` and `max.poll.interval.ms` values in the configuration tensor
  and try again after the group rebalances. For example: considering the kafka cluster
  has been setup with the default settings, `max.poll.interval.ms` would be `300000ms`.
  It can be changed to `8000ms` to reduce the time between pools. Also, the `session.timeout.ms`
  can be changed to `7000ms`. However, the value for `session.timeout.ms` should be
  according to the following relation:

  - `group.max.session.timeout.ms` in server.properties > `session.timeout.ms` in the
  consumer.properties.
  - `group.min.session.timeout.ms` in server.properties < `session.timeout.ms` in the
  consumer.properties

  >>> dataset = tfio.experimental.streaming.KafkaGroupIODataset(
                      topics=["topic1"],
                      group_id="cg",
                      servers="localhost:9092",
                      configuration=[
                          "session.timeout.ms=7000",
                          "max.poll.interval.ms=8000",
                          "auto.offset.reset=earliest",
                      ],
                  )

  In the above example, the `auto.offset.reset` configuration is set to `earliest` so that
  in case the consumer group is being newly created, it will start reading the messages from
  the beginning. If it is not set, it defaults to `latest`. For additional configurations,
  please refer the librdkafka's configurations:
  https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md

  In addition to the standard streaming functionality, there is added support for a timeout
  based stream. Once the existing data has been fetched, this dataset will block for
  an additional `stream_timeout` milliseconds, for the new messages to be captured.

  >>> dataset = tfio.experimental.streaming.KafkaGroupIODataset(
                      topics=["topic1"],
                      group_id="cg",
                      servers="localhost:9092",
                      stream_timeout=30000,
                      configuration=[
                          "session.timeout.ms=7000",
                          "max.poll.interval.ms=8000",
                          "auto.offset.reset=earliest",
                      ],
                  )
  >>> for (message, key) in dataset:
  ...     print(message)

  The above loop will run as long as the consumer clients are able to fetch messages
  from the topic(s). However, since we set the `stream_timeout` value to `15000` milliseconds,
  the dataset will wait for any new messages that might be added to the topic for that duration.

  As the kafka deployments vary in configuration as per various use-cases, the time required for
  the consumers to fetch a single message might also vary. This timeout value can be adjusted
  using the `message_poll_timeout` parameter.

  The `message_poll_timeout` value represents the duration which the consumers
  have to wait while fetching a new message. However, even if we receive a new message
  before the `message_poll_timeout` interval finishes, the consumer doesn't resume the
  consumption but it will wait until the `message_poll_timeout` interval has finished.
  Thus, if we want to block indefinitely until a new message arrives,
  we cannot do it with `message_poll_timeout` alone. This is when the `stream_timeout`
  value comes in, where we can set the value to a very high timeout
  (i.e, block indefinitely) and keep on polling for new messages at
  `message_poll_timeout` intervals.
  """

  def __init__(
    self,
    topics,
    group_id,
    servers,
    stream_timeout=0,
    message_poll_timeout=10000,
    configuration=None,
    internal=True,
  ):
    """
    Args:
      topics: A `tf.string` tensor containing topic names in [topic] format.
        For example: ["topic1", "topic2"]
      group_id: The id of the consumer group. For example: cgstream
      servers: An optional list of bootstrap servers.
        For example: `localhost:9092`.
      stream_timeout: An optional timeout duration (in milliseconds) to block until
        the new messages from kafka are fetched.
        By default it is set to 0 milliseconds and doesn't block for new messages.
        To block indefinitely, set it to -1.
      message_poll_timeout: An optional timeout duration (in milliseconds)
        after which the kafka consumer throws a timeout error while fetching
        a single message. This value also represents the intervals at which
        the kafka topic(s) are polled for new messages while using the `stream_timeout`
      configuration: An optional `tf.string` tensor containing
        configurations in [Key=Value] format.
        Global configuration: please refer to 'Global configuration properties'
          in librdkafka doc. Examples include
          ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
        Topic configuration: please refer to 'Topic configuration properties'
          in librdkafka doc. Note all topic configurations should be
          prefixed with `conf.topic.`. Examples include
          ["conf.topic.auto.offset.reset=earliest"]
        Reference: https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md
      internal: Whether the dataset is being created from within the named scope.
        Default: True
    """
    with tf.name_scope("KafkaGroupIODataset"):
      assert internal

      if stream_timeout == -1:
        stream_timeout = sys.maxsize
      elif stream_timeout >= 0:
        # Taking the max of `stream_timeout` and `message_poll_timeout`
        # to prevent the user from bothering about the underlying polling
        # mechanism.
        stream_timeout = max(stream_timeout, message_poll_timeout)
      else:
        raise ValueError("Invalid stream_timeout value: {} ,set it to -1 to block indefinitely.".format(stream_timeout))
      metadata = list(configuration or [])
      if group_id is not None:
        metadata.append("group.id=%s" % group_id)
      if servers is not None:
        metadata.append("bootstrap.servers=%s" % servers)
      resource = core_ops.io_kafka_group_readable_init(topics=topics, metadata=metadata)

      self._resource = resource
      dataset = tf.data.Dataset.counter()
      dataset = dataset.map(
        lambda i: core_ops.io_kafka_group_readable_next(
          input=self._resource,
          index=i,
          message_poll_timeout=message_poll_timeout,
          stream_timeout=stream_timeout,
        )
      )
      dataset = dataset.take_while(lambda v: tf.greater(v.continue_fetch, 0))
      dataset = dataset.map(lambda v: v.message)
      dataset = dataset.unbatch()

      self._dataset = dataset
      super().__init__(self._dataset._variant_tensor)  # pylint: disable=protected-access

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._dataset.element_spec


class KafkaPipeline(DataPipeline, ABC):
  def __init__(
    self,
    topics,
    group_id,
    servers,
    stream_timeout=None,
    configuration=None,
    compression_type=None,
    num_client=1,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.topics = topics
    self.group_id = group_id
    self.servers = servers
    self.stream_timeout = stream_timeout
    self.configuration = configuration
    self.compression_type = compression_type
    self.num_client = num_client

  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    if self.num_client > 1:
      logger.info(f"Using {self.num_client} Kafka clients.")
      clients = tuple(
        KafkaGroupIODataset(
          topics=self.topics,
          group_id=self.group_id,
          servers=self.servers,
          stream_timeout=self.stream_timeout,
          configuration=self.configuration,
        )
        for _ in range(self.num_client)
      )
      dataset = tf.data.Dataset.zip(clients)
      dataset = dataset.map(lambda *x: tf.stack(x, axis=-1)).unbatch()
    else:
      dataset = KafkaGroupIODataset(
        topics=self.topics,
        group_id=self.group_id,
        servers=self.servers,
        stream_timeout=self.stream_timeout,
        configuration=self.configuration,
      )
      logger.info(
        "Using only one Kafka client, if there is an IO bottleneck, it is recommended to adjust 'num_client' to increase the number of Kafka clients"
      )
    if self.prebatch_size:
      if batch_size > self.prebatch_size:
        dataset = dataset.batch(
          batch_size=batch_size // self.prebatch_size,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=True,
          drop_remainder=True,
        )
    else:
      dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True, drop_remainder=True)
    if self.compression_type:
      dataset = dataset.map(
        lambda v: tf.io.decode_compressed(v, compression_type=self.compression_type), multiprocessing.cpu_count()
      )
    if not hasattr(self.parser, "__isabstractmethod__"):
      dataset = dataset.map(self.parser, multiprocessing.cpu_count())
    if self.prebatch_size and batch_size % self.prebatch_size != 0:
      dataset = dataset.unbatch().batch(batch_size)
    return dataset
