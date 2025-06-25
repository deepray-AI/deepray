import os

import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def sparse_faeture_dict(feat_name, feat_num, dtype, ftype, embed_dim=4, length=1):
  """
  为每个离散变量建立信息字典
  :param feat_name: 特征名。
  :param feat_num: 特征数，每一个特征编码后对应有多少个类别。
  :param embed_dim: 特征维度，特征embedding后的维度。
  :return:
  """
  return {"name": feat_name, "voc_size": feat_num, "dim": embed_dim, "length": length, "dtype": dtype, "ftype": ftype}


def create_avazu_dataset(path, out_path, read_part=False, samples_num=100, embed_dim=8):
  print("数据预处理开始")
  sparse_features = [
    "hour",
    "id",
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
  ]

  train_path = os.path.join(path + "/train.gz")
  test_path = os.path.join(path + "/test.gz")
  print("加载数据集")
  if read_part:
    train_data = pd.read_csv(train_path, nrows=samples_num)
    test_x = pd.read_csv(test_path, nrows=samples_num)
  else:
    train_data = pd.read_csv(train_path)
    test_x = pd.read_csv(test_path)
  print(train_data.head())

  print(test_x.head())

  # hour, 只有14年10月11天的数据，year,month没必要做特征
  train_data["hour"] = train_data["hour"].apply(str)
  train_data["hour"] = train_data["hour"].map(lambda x: int(x[6:8]))  # int强转去掉字符串前的0
  test_x["hour"] = test_x["hour"].apply(str)
  test_x["hour"] = test_x["hour"].map(lambda x: int(x[6:8]))
  print("加载数据完成")
  print("Sparse feature encode")
  # sparse features
  le = LabelEncoder()
  for feat in sparse_features:
    all_class = pd.concat([train_data[feat], test_x[feat]]).unique()
    print(f"Processing {feat}, count {len(all_class)}")
    le.fit(all_class)
    train_data[feat] = le.transform(train_data[feat])
    test_x[feat] = le.transform(test_x[feat])

  print("Sparse feature encode succeed")
  # save LabelEncoder model for test
  # joblib.dump(le, 'label_encoder.model')
  # sparse_faeture_dict(feat_name='day', feat_num=32, embed_dim=embed_dim)
  # sparse_faeture_dict(feat_name='hour', feat_num=24, embed_dim=embed_dim)

  train_data = train_data[sparse_features + ["click"]].astype("int32")

  train, val = train_test_split(train_data, test_size=0.2, shuffle=True)

  test_x = test_x[sparse_features].astype("int32")

  features_columns = [
    sparse_faeture_dict(
      feat_name=feat,
      feat_num=train_data[feat].max() + 1,
      dtype=train_data[feat].dtype,
      ftype="Categorical",
      embed_dim=embed_dim,
    )
    for feat in sparse_features
  ]

  feature_map = pd.DataFrame(features_columns)
  feature_map.loc[len(feature_map)] = {
    "name": "click",
    "dtype": "int32",
    "ftype": "Label",
    "length": 1,
    "voc_size": 1,
    "dim": 1,
  }
  feature_map = feature_map.astype({
    "length": int,
    "voc_size": int,
    "dim": int,
  })

  print(feature_map)

  feature_map.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "feature_map.csv"), index=False)

  train.to_parquet(os.path.join(out_path + "train.parquet"))
  val.to_parquet(os.path.join(out_path + "valid.parquet"))
  test_x.to_parquet(os.path.join(out_path + "test.parquet"))

  print("Data preprocessing completed.")


if __name__ == "__main__":
  path = "/workspaces/dataset/avazu/"
  out_path = "/workspaces/dataset/avazu/output/"

  if not os.path.exists(out_path):
    os.makedirs(out_path)

  embed_dim = 11

  create_avazu_dataset(
    path,
    out_path,
    read_part=False,
    embed_dim=embed_dim,
  )
