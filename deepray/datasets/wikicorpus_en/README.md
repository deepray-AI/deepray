## Quick Start Guide
 

 
Required data is downloaded into the `data/` directory by default.
 
1. Download and preprocess the dataset.
 
This repository provides scripts to download, verify, and extract the following datasets:
 
-   Wikipedia (pre-training)
 
To download, verify, extract the datasets, and create the shards in `tfrecord` format, run:
```
export DATA_PREP_WORKING_DIR=/workspaces/dataset/wikicorpus_en/data
bash create_datasets_from_start.sh wiki_only
```

The processing scripts colletced from https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/LanguageModeling/ELECTRA/README.md#quick-start-guide