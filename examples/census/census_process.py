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

train_df, valid_df = train_test_split(df, test_size=0.2)
converter(train_df, out_file='./data/train.tfrecord', prebatch=prebatch)
converter(valid_df, out_file='./data/valid.tfrecord', prebatch=prebatch)
