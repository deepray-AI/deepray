# Quick-Start

## Installation
DeePray is currently hosted on `PyPI <https://pypi.org/project/deepray/>`. You can simply install DeePray with the following command:

    pip3 install deepray

You can also install with the newest version through GitHub:

    pip3 install git+https://github.com/fuhailin/deepray.git@master

If you use Anaconda or Miniconda, you can install DeePray through the following command lines:

    # create a new virtualenv and install pip, change the env name if you like
    conda create -n myenv pip
    # activate the environment
    conda activate myenv
    # install deepray
    pip install deepray

After installation, open your python console and type

    import deepray as dp
    print(dp.__version__)

If no error occurs, you have successfully installed DeePray.
## Getting started: 2 steps to DeePray


### Step 1: Process Data


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from deepray.utils.converter import CSV2TFRecord

LABEL = ['income_label']
train_data = '/Users/vincent/Documents/projects/DeePray/examples/census/data/raw_data/adult_data.csv'
df = pd.read_csv(train_data)
df['income_label'] = (df["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df.pop('income_bracket')

NUMERICAL_FEATURES = ['age', 'fnlwgt', 'hours_per_week', 'capital_gain', 'capital_loss', 'education_num']
CATEGORY_FEATURES = [col for col in df.columns if col != LABEL and col not in NUMERICAL_FEATURES]

for feat in CATEGORY_FEATURES:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])
# Feature normilization
mms = MinMaxScaler(feature_range=(0, 1))
df[NUMERICAL_FEATURES] = mms.fit_transform(df[NUMERICAL_FEATURES])
# target = df.pop(LABEL)

prebatch = 1  # flags.prebatch
converter = CSV2TFRecord(LABEL, NUMERICAL_FEATURES, CATEGORY_FEATURES, VARIABLE_FEATURES=[], gzip=True)
converter.write_feature_map(df, './data/feature_map.csv')
```
Your feature map file should look like that:

```
9,workclass,CATEGORICAL
16,education,CATEGORICAL
7,marital_status,CATEGORICAL
15,occupation,CATEGORICAL
6,relationship,CATEGORICAL
5,race,CATEGORICAL
2,gender,CATEGORICAL
42,native_country,CATEGORICAL
1,age,NUMERICAL
1,fnlwgt,NUMERICAL
1,hours_per_week,NUMERICAL
1,capital_gain,NUMERICAL
1,capital_loss,NUMERICAL
1,education_num,NUMERICAL
2,income_label,LABEL
```
And then save DataFrame as TFRecord format file, I have provided a tool to do this process, 
just `from deepray.utils.converter import CSV2TFRecord`
```python
train_df, valid_df = train_test_split(df, test_size=0.2)
converter(train_df, out_file='./data/train.tfrecord', prebatch=prebatch)
converter(valid_df, out_file='./data/valid.tfrecord', prebatch=prebatch)
```
    


### Step 2: Config your model

```python
import sys
import time

from absl import app, flags

from deepray.base.trainer import train
from deepray.model.build_model import BuildModel

FLAGS = flags.FLAGS


def main(unused=None, flags=None):
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

```






