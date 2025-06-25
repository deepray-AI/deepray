import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from deepray.custom_ops.parquet_dataset import parquet_dataset_ops

ROW = 10

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_workspace = tempfile.mkdtemp()
_filename = os.path.join(_workspace, "test.parquet")
print(_filename)
# _df = pd.DataFrame(
#     np.random.randint(0, 100, size=(200, 4), dtype=np.int64),
# columns=list('ABCd'))

_df = pd.DataFrame({
  "A": [np.random.randint(0, 10, size=np.random.randint(0, 5), dtype="int64").tolist() for _ in range(ROW)],
  "B": [np.random.randint(0, 10, size=3, dtype="int64").tolist() for _ in range(ROW)],
  "C": np.random.randint(0, 100, size=ROW, dtype="int32"),
  "D": np.random.randint(0, 1000, size=ROW, dtype="int64"),
  "E": ["string_{}".format(i) for i in range(ROW)],
})

print(_df.head(10))

_df.to_parquet(_filename)

batch_size = 5
ds = parquet_dataset_ops.ParquetDataset(
  _filename,
  batch_size=batch_size,
  fields=["A", "C"],
  # fields=[
  #     parquet_dataset_ops.DataFrame.Field('A', tf.int64, ragged_rank=1),
  #     parquet_dataset_ops.DataFrame.Field(
  #         'B',
  #         tf.int64,
  #         shape=[3],
  #     ),
  #     parquet_dataset_ops.DataFrame.Field('C', tf.int32),
  #     parquet_dataset_ops.DataFrame.Field('D', tf.int64),
  #     parquet_dataset_ops.DataFrame.Field('E', tf.string),
  # ]
)

ds = ds.prefetch(4)
for x in ds:
  for name, tensor in x.items():
    print(f"{name}: {tensor}, type = {type(tensor)}")
  print("\n")
