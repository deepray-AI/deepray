import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

src_file = "taobao/taobao_test_data"
dst_file = src_file + ".parquet"

chunksize = 10000  # this is the number of lines

# Definition of some constants
LABEL_COLUMNS = ["clk", "buy"]
HASH_INPUTS = [
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
]
IDENTITY_INPUTS = ["price"]
ALL_INPUT = LABEL_COLUMNS + HASH_INPUTS + IDENTITY_INPUTS

label_dtype = {item: int for item in LABEL_COLUMNS}
hash_dtype = {item: str for item in HASH_INPUTS}
identity_dtype = {item: int for item in IDENTITY_INPUTS}
input_dtype = {}
input_dtype.update(label_dtype)
input_dtype.update(hash_dtype)
input_dtype.update(identity_dtype)

label_field = [pa.field(item, pa.int32()) for item in LABEL_COLUMNS]
hash_field = [pa.field(item, pa.string()) for item in HASH_INPUTS]
identity_field = [pa.field(item, pa.int32()) for item in IDENTITY_INPUTS]
input_field = label_field + hash_field + identity_field

label_default_values = {item: 0 for item in LABEL_COLUMNS}
hash_default_values = {item: " " for item in HASH_INPUTS}
identity_default_values = {item: 0 for item in IDENTITY_INPUTS}
default_values = {}
default_values.update(label_default_values)
default_values.update(hash_default_values)
default_values.update(identity_default_values)

schema = pa.schema(input_field)

pqwriter = pq.ParquetWriter(dst_file, schema)
for i, df in enumerate(pd.read_csv(src_file, chunksize=chunksize, names=ALL_INPUT, dtype=input_dtype)):
  df = df.fillna(default_values)
  table = pa.Table.from_pandas(df, schema)
  pqwriter.write_table(table)

# close the parquet writer
if pqwriter:
  pqwriter.close()

output_table = pq.read_table(dst_file)
print(output_table.to_pandas())
