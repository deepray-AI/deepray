import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

src_file = "taobao/taobao_test_data"
dst_file = src_file + ".parquet"

chunksize = 10000  # this is the number of lines

# Definition of some constants
INPUT_FEATURES = [
  "pid",
  "adgroup_id",
  "cate_id",
  "campaign_id",
  "customer",
  "brand",
  "user_id",
  "cms_segid",
  "cms_group_id",
  "final_gender_code",
  "age_level",
  "pvalue_level",
  "shopping_level",
  "occupation",
  "new_user_class_level",
  "tag_category_list",
  "tag_brand_list",
  "price",
]
LABEL_COLUMN = ["clk"]
BUY_COLUMN = ["buy"]
INPUT_COLUMN = LABEL_COLUMN + BUY_COLUMN + INPUT_FEATURES

label_dtype = {label: int for label in LABEL_COLUMN}
buy_dtype = {buy: int for buy in BUY_COLUMN}
features_dtype = {feature: str for feature in INPUT_FEATURES}
input_dtype = {}
input_dtype.update(label_dtype)
input_dtype.update(buy_dtype)
input_dtype.update(features_dtype)

label_field = [pa.field(label, pa.int32()) for label in LABEL_COLUMN]
buy_field = [pa.field(buy, pa.int32()) for buy in BUY_COLUMN]
features_field = [pa.field(feature, pa.string()) for feature in INPUT_FEATURES]
input_field = label_field + buy_field + features_field

label_default_values = {label: 0 for label in LABEL_COLUMN}
buy_default_values = {buy: 0 for buy in BUY_COLUMN}
features_default_values = {feature: " " for feature in INPUT_FEATURES}
default_values = {}
default_values.update(label_default_values)
default_values.update(buy_default_values)
default_values.update(features_default_values)

schema = pa.schema(input_field)

pqwriter = pq.ParquetWriter(dst_file, schema)
for i, df in enumerate(pd.read_csv(src_file, chunksize=chunksize, names=INPUT_COLUMN, dtype=input_dtype)):
  df = df.fillna(default_values)
  table = pa.Table.from_pandas(df, schema)
  pqwriter.write_table(table)

# close the parquet writer
if pqwriter:
  pqwriter.close()

output_table = pq.read_table(dst_file)
print(output_table.to_pandas())
