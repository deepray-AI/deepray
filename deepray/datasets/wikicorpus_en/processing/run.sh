DATASET="wikicorpus_en"
# Properly format the text files
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action text_formatting --dataset wikicorpus_en

# Shard the text files
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action sharding --dataset $DATASET

# Create TFRecord files Phase 1
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 128 \
--max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt


# Create TFRecord files Phase 2
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset ${DATASET} --max_seq_length 512 \
--max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt
