DATASET="wikicorpus_en"
BERT_PREP_WORKING_DIR="/workspaces/deepray/deepray/datasets/openwebtext"

# Properly format the text files
python3 bertPrep.py --action text_formatting --dataset wikicorpus_en

# Shard the text files
python3 bertPrep.py --action sharding --dataset $DATASET

# Create TFRecord files Phase 1
python3 bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 128 \
--max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/vocab.txt


# Create TFRecord files Phase 2
python3 bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/vocab.txt
