# Shard the raw training/test data into multiple files
python3 shard_rebalancer.py \
  --input_path "/workspaces/dataset/criteo_raw/train.csv" \
  --output_path "/workspaces/dataset/criteo/train/" \
  --num_output_files 1 --filetype csv --runner DirectRunner

python3 shard_rebalancer.py \
  --input_path "/workspaces/dataset/criteo/dac/criteo_raw/test.csv" \
  --output_path "/workspaces/dataset/criteo/dac/test/" \
  --num_output_files 5 --filetype csv --runner DirectRunner
#  --project ${PROJECT} --region ${REGION}

# Generate vocabulary and preprocess the data.
python3 criteo_preprocess.py \
  --input_path "/workspaces/dataset/criteo_raw/*" \
  --output_path "/workspaces/dataset/criteo/dac/output/" \
  --csv_delimeter "," \
  --temp_dir "/workspaces/dataset/criteo/dac/criteo_vocab/" \
  --vocab_gen_mode --runner DirectRunner --max_vocab_size 5000000

# Preprocess training and test data:
python3 criteo_preprocess.py \
  --input_path "/workspaces/dataset/criteo/dac/criteo_raw/train.txt" \
  --output_path "/workspaces/dataset/criteo/train/" \
  --temp_dir "/workspaces/dataset/criteo/dac/criteo_vocab/" \
  --runner DirectRunner --max_vocab_size 5000000

python3 criteo_preprocess.py \
  --input_path "/workspaces/dataset/criteo/dac/criteo_raw/test.csv" \
  --output_path "/workspaces/dataset/criteo/test/" \
  --temp_dir "/workspaces/dataset/criteo/dac/criteo_vocab/" \
  --runner DirectRunner --max_vocab_size 5000000


curl https://sacriteopcail01.z16.web.core.windows.net/day_0.gz | hadoop fs -appendToFile - hdfs://10.11.11.241:8020/data/day_0.gz
aria2c -x 16 https://sacriteopcail01.z16.web.core.windows.net/day_0.gz
hadoop fs -cat hdfs://10.11.11.241:8020/data/day_0.gz | gzip -d | hadoop fs -put - hdfs://10.11.11.241:8020/data/day_0


for i in `seq 0 23`
do
  echo "output: $i"
  aria2c -x 16 https://sacriteopcail01.z16.web.core.windows.net/day_$i.gz
  mc cp day_$i.gz minio/datasets/criteo/
done


python3 shard_rebalancer.py \
  --input_path "/workspaces/dataset/criteo/dac/test.parquet" \
  --output_path "/workspaces/dataset/criteo/dac/test/" \
  --num_output_files 5 --filetype parquet --runner DirectRunner