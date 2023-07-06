
#### Setup
1. Place a vocabulary file in `$DATA_DIR/vocab.txt`. Our ELECTRA models all used the exact same vocabulary as English uncased BERT, which you can download [here](https://storage.googleapis.com/electra-data/vocab.txt).
2. Download the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) corpus (12G) and extract it  (i.e., run `tar xf openwebtext.tar.xz`). Place it in `$DATA_DIR/openwebtext`.
3. Run `python3 build_openwebtext_pretraining_dataset.py --data-dir $DATA_DIR --num-processes 5`. It pre-processes/tokenizes the data and outputs examples as [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files under `$DATA_DIR/pretrain_tfrecords`. The tfrecords require roughly 30G of disk space.
