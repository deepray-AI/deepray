**DeePray** (`深度祈祷`): A new Modular, Scalable, Configurable, Easy-to-Use and Extend infrastructure for Deep Learning based Recommendation.

[![Documentation Status](https://readthedocs.org/projects/deepray/badge/?version=latest)](https://deepray.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/deepray.svg)](https://badge.fury.io/py/deepray)
[![GitHub version](https://badge.fury.io/gh/fuhailin%2Fdeepray.svg)](https://badge.fury.io/gh/fuhailin%2Fdeepray)


## Introduction
The DeePray library offers state-of-the-art algorithms for [deep learning recommendation].
DeePray is built on latest [TensorFlow 2][(https://tensorflow.org/)] and designed with modular structure，
making it easy to discover patterns and answer questions about tabular-structed data.

The main goals of DeePray:

- Easy to use, newbies can get hands dirty with deep learning quickly
- Good performance with web-scale data
- Easy to extend, Modular architecture let you build your Neural network like playing LEGO!

Let's Get Started! Please refer to the official docs at https://deepray.readthedocs.io/en/latest/.

## Installation


#### Install DeePray using PyPI:

To install DeePray library from [PyPI](https://pypi.org/) using `pip`, execute the following command:

```
pip install deepray
```

#### Install DeePray from Github source:

First, clone the DeePray repository using `git`:

```
git clone https://github.com/fuhailin/deepray.git
```

Then, `cd` to the deepray folder, and install the library by executing the following commands:

```
cd deepray
pip install .
```
## Tutorial

### Census Adult Data Set
#### Data preparation

In your tabular data, specify **NUMERICAL** for your continue features,  **CATEGORY** for categorical features, **VARIABLE** for variable length features, and obviously **LABEL** for label column. Then process them to  to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format into order to get good performance with large-scale dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from deepray.utils.converter import CSV2TFRecord


# http://archive.ics.uci.edu/ml/datasets/Adult
train_data = 'DeePray/examples/census/data/raw_data/adult_data.csv'
df = pd.read_csv(train_data)
df['income_label'] = (df["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df.pop('income_bracket')

NUMERICAL_FEATURES = ['age', 'fnlwgt', 'hours_per_week', 'capital_gain', 'capital_loss', 'education_num']
CATEGORY_FEATURES = [col for col in df.columns if col != LABEL and col not in NUMERICAL_FEATURES]
LABEL = ['income_label']

for feat in CATEGORY_FEATURES:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])
# Feature normilization
mms = MinMaxScaler(feature_range=(0, 1))
df[NUMERICAL_FEATURES] = mms.fit_transform(df[NUMERICAL_FEATURES])


prebatch = 1  # flags.prebatch
converter = CSV2TFRecord(LABEL, NUMERICAL_FEATURES, CATEGORY_FEATURES, VARIABLE_FEATURES=[], gzip=False)
converter.write_feature_map(df, './data/feature_map.csv')

train_df, valid_df = train_test_split(df, test_size=0.2)
converter(train_df, out_file='./data/train.tfrecord', prebatch=prebatch)
converter(valid_df, out_file='./data/valid.tfrecord', prebatch=prebatch)
```

You will get a feature map file like that:

```
9,workclass,CATEGORICAL
16,education,CATEGORICAL
7,marital_status,CATEGORICAL
15,occupation,CATEGORICAL
6,relationship,CATEGORICAL
5,race,CATEGORICAL
2,gender,CATEGORICAL
42,native_country,CATEGORICAL
1,hours_per_week,NUMERICAL
1,capital_gain,NUMERICAL
1,age,NUMERICAL
1,fnlwgt,NUMERICAL
1,capital_loss,NUMERICAL
1,education_num,NUMERICAL
2,income_label,LABEL
```
And then create two txt file `train`and `valid` separately to record train set TFRecords and valid set TFRecords file path.


### Choose your model, Training and evaluation

```python
"""
build and train model
"""

import sys

from absl import app, flags

import deepray as dp
from deepray.base.trainer import train
from deepray.model.build_model import BuildModel

FLAGS = flags.FLAGS


def main(flags=None):
    FLAGS(flags, known_only=True)
    flags = FLAGS
    model = BuildModel(flags)
    history = train(model)
    print(history)


argv = [
    sys.argv[0],
    '--model=lr',
    '--train_data=/Users/vincent/Projects/DeePray/examples/census/data/train',
    '--valid_data=/Users/vincent/Projects/DeePray/examples/census/data/valid',
    '--feature_map=/Users/vincent/Projects/DeePray/examples/census/data/feature_map.csv',
    '--learning_rate=0.01',
    '--epochs=10',
    '--batch_size=64',
]
main(flags=argv)
```


## Models List

| Titile                                                       |  Booktitle  | Resources                                                    |
| ------------------------------------------------------------ | :---------: | ------------------------------------------------------------ |
| **FM**: Factorization Machines                               |  ICDM'2010  | [[pdf]](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_fm.py) |
| **FFM**: Field-aware Factorization Machines for CTR Prediction | RecSys'2016 | [[pdf]](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_ffm.py) |
| **FNN**: Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction |  ECIR'2016  | [[pdf]](https://arxiv.org/abs/1601.02376)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_fnn.py) |
| **PNN**: Product-based Neural Networks for User Response Prediction |  ICDM'2016  | [[pdf]](https://arxiv.org/abs/1611.00144)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_pnn.py) |
| **Wide&Deep**: Wide & Deep Learning for Recommender Systems  |  DLRS'2016  | [[pdf]](https://arxiv.org/pdf/1606.07792)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_wdl.py) |
| **AFM**: Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | IJCAI'2017  | [[pdf]](https://arxiv.org/abs/1708.04617)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_afm.py) |
| **NFM**: Neural Factorization Machines for Sparse Predictive Analytics | SIGIR'2017  | [[pdf]](https://arxiv.org/abs/1708.05027)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_nfm.py) |
| **DeepFM**: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[C] | IJCAI'2017  | [[pdf]](https://arxiv.org/abs/1703.04247) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_deepfm.py) |
| **DCN**: Deep & Cross Network for Ad Click Predictions       | ADKDD'2017  | [[pdf]](https://arxiv.org/abs/1708.05123) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dcn.py) |
| **xDeepFM**: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems |  KDD'2018   | [[pdf]](https://arxiv.org/abs/1803.05170) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_xdeepfm.py) |
| **DIN**: DIN: Deep Interest Network for Click-Through Rate Prediction |  KDD'2018   | [[pdf]](https://arxiv.org/abs/1706.06978) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dien.py) |
| **DIEN**: DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction |  AAAI'2019  | [[pdf]](https://arxiv.org/abs/1809.03672) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dien.py) |
| **DSIN**: Deep Session Interest Network for Click-Through Rate Prediction | IJCAI'2019  | [[pdf]](https://arxiv.org/abs/1905.06482)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dsin.py) |
| **AutoInt**: Automatic Feature Interaction Learning via Self-Attentive Neural Networks |  CIKM'2019  | [[pdf]](https://arxiv.org/abs/1810.11921)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_autoint.py) |
| **FLEN**: Leveraging Field for Scalable CTR Prediction       |  AAAI'2020  | [[pdf]](https://arxiv.org/pdf/1911.04690.pdf)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_flen.py) |
| **DFN**: Deep Feedback Network for Recommendation            | IJCAI'2020  | [[pdf]]()[[code]](TODO)                                      |

# How to build your own model with DeePray

Inheriting   `BaseCTRModel` class from `from deepray.model.model_ctr`, and implement your own `build_network()` method!


# Contribution

DeePray is still under development, and call for contributions!

```
* Hailin Fu (`Hailin <https://github.com/fuhailin>`)
* Call for contributions!
```
让DeePray成为推荐算法新基建需要你的贡献

# Citing
DeePray is designed, developed and supported by [Hailin](https://github.com/fuhailin/).
If you use any part of this library in your research, please cite it using the following BibTex entry
```latex
@misc{DeePray,
  author = {Hailin Fu},
  title = {DeePray: A new Modular, Scalable, Configurable, Easy-to-Use and Extend infrastructure for Deep Learning based Recommendation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/fuhailin/deepray}},
}
```

# License

Copyright (c) Copyright © 2020 The DeePray Authors<Hailin Fu>. All Rights Reserved.

Licensed under the [Apach](LICENSE) License.

# Reference

https://github.com/shenweichen/DeepCTR

https://github.com/aimetrics/jarvis

https://github.com/shichence/AutoInt

# Contact
If you want cooperation or have any questions, please follow my wechat offical account:

公众微信号ID：【StateOfTheArt】

![StateOfTheArt](https://gitee.com/fuhailin/Object-Storage-Service/raw/master/wechat_channel.png)