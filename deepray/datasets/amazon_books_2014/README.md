# SIM For TensorFlow 2

This repository provides a script and recipe to train the SIM model to achieve state-of-the-art accuracy. The content of this repository is tested and maintained by NVIDIA.

## Table Of Contents

- [Model overview](#model-overview)
    * [Model architecture](#model-architecture)
    * [Default configuration](#default-configuration)
    * [Feature support matrix](#feature-support-matrix)
	    * [Features](#features)
    * [Mixed precision training](#mixed-precision-training)
	    * [Enabling mixed precision](#enabling-mixed-precision)
          * [Enabling TF32](#enabling-tf32)
    * [BYO dataset functionality overview](#byo-dataset-functionality-overview)
        * [BYO dataset glossary](#byo-dataset-glossary)
        * [Dataset feature specification](#dataset-feature-specification)
        * [Data flow in NVIDIA Deep Learning Examples recommendation models](#data-flow-in-nvidia-deep-learning-examples-recommendation-models)
        * [Example of dataset feature specification](#example-of-dataset-feature-specification)
        * [BYO dataset functionality](#byo-dataset-functionality)
    * [Glossary](#glossary)
- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
        * [Dataset guidelines](#dataset-guidelines)
        * [Prebatching](#prebatching)
        * [BYO dataset](#byo-dataset)
            * [Channel definitions and requirements](#channel-definitions-and-requirements)
    * [Training process](#training-process)
    * [Inference process](#inference-process)
    * [Log format](#log-format)
        * [Training log data](#training-log-data)
        * [Inference log data](#inference-log-data)
- [Performance](#performance)
    * [Benchmarking](#benchmarking)
        * [Training performance benchmark](#training-performance-benchmark)
        * [Inference performance benchmark](#inference-performance-benchmark)
    * [Results](#results)
        * [Training accuracy results](#training-accuracy-results)                         
            * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)  
            * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
            * [Training stability test](#training-stability-test)
            * [Impact of mixed precision on training accuracy](#impact-of-mixed-precision-on-training-accuracy)
            * [Training accuracy plot](#training-accuracy-plot)
        * [Training performance results](#training-performance-results)
            * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
            * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
        * [Inference performance results](#inference-performance-results)
            * [Inference performance: NVIDIA DGX A100 (8x A100 80GB)](#inference-performance-nvidia-dgx-a100-8x-a100-80gb)
            * [Inference performance: NVIDIA DGX-2 (16x V100 32GB)](#inference-performance-nvidia-dgx-2-16x-v100-32gb)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)


### BYO dataset functionality overview

This section describes how you can train the DeepLearningExamples RecSys models on your own datasets without changing
the model or data loader and with similar performance to the one published in each repository.
This can be achieved thanks to Dataset Feature Specification, which describes how the dataset, data loader, and model
interact with each other during training, inference, and evaluation.
Dataset Feature Specification has a consistent format across all recommendation models in NVIDIA’s DeepLearningExamples
repository, regardless of dataset file type and the data loader,
giving you the flexibility to train RecSys models on your own datasets.

- [BYO dataset glossary](#byo-dataset-glossary)
- [Dataset Feature Specification](#dataset-feature-specification)
- [Data Flow in Recommendation Models in DeepLearning examples](#data-flow-in-nvidia-deep-learning-examples-recommendation-models)
- [Example of Dataset Feature Specification](#example-of-dataset-feature-specification)
- [BYO dataset functionality](#byo-dataset-functionality)

#### BYO dataset glossary

The Dataset Feature Specification consists of three mandatory and one optional section:

<b>feature_spec </b> provides a base of features that may be referenced in other sections, along with their metadata.
	Format: dictionary (feature name) => (metadata name => metadata value)<br>

<b>source_spec </b> provides information necessary to extract features from the files that store them. 
	Format: dictionary (mapping name) => (list of chunks)<br>

* <i>Mappings</i> are used to represent different versions of the dataset (think: train/validation/test, k-fold splits). A mapping is a list of chunks.<br>
* <i>Chunks</i> are subsets of features that are grouped together for saving. For example, some formats may constrain data saved in one file to a single data type. In that case, each data type would correspond to at least one chunk. Another example where this might be used is to reduce file size and enable more parallel loading. Chunk description is a dictionary of three keys:<br>
  * <i>type</i> provides information about the format in which the data is stored. Not all formats are supported by all models.<br>
  * <i>features</i> is a list of features that are saved in a given chunk. The order of this list may matter: for some formats, it is crucial for assigning read data to the proper feature.<br>
  * <i>files</i> is a list of paths to files where the data is saved. For Feature Specification in yaml format, these paths are assumed to be relative to the yaml file’s directory (basename). <u>Order of this list matters:</u> It is assumed that rows 1 to i appear in the first file, rows i+1 to j in the next one, etc. <br>

<b>channel_spec</b> determines how features are used. It is a mapping (channel name) => (list of feature names). 

Channels are model-specific magic constants. In general, data within a channel is processed using the same logic. Example channels: model output (labels), categorical ids, numerical inputs, user data, and item data.

<b>metadata</b> is a catch-all, wildcard section: If there is some information about the saved dataset that does not fit into the other sections, you can store it here.

#### Dataset feature specification

Data flow can be described abstractly:
Input data consists of a list of rows. Each row has the same number of columns; each column represents a feature.
The columns are retrieved from the input files, loaded, aggregated into channels and supplied to the model/training script. 

FeatureSpec contains metadata to configure this process and can be divided into three parts:

* Specification of how data is organized on disk (source_spec). It describes which feature (from feature_spec) is stored in which file and how files are organized on disk.

* Specification of features (feature_spec). Describes a dictionary of features, where key is the feature name and values are the features’ characteristics such as  dtype and other metadata (for example, cardinalities for categorical features)

* Specification of model’s inputs and outputs (channel_spec). Describes a dictionary of model’s inputs where keys specify model channel’s names and values specify lists of features to be loaded into that channel. Model’s channels are groups of data streams to which common model logic is applied, for example categorical/continuous data, and user/item ids. Required/available channels depend on the model


The FeatureSpec is a common form of description regardless of underlying dataset format, dataset data loader form, and model. 


#### Data flow in NVIDIA Deep Learning Examples recommendation models

The typical data flow is as follows:
* <b>S.0.</b> Original dataset is downloaded to a specific folder.
* <b>S.1.</b> Original dataset is preprocessed into Intermediary Format. For each model, the preprocessing is done differently, using different tools. The Intermediary Format also varies (for example, for DLRM PyTorch, the Intermediary Format is a custom binary one.)
* <b>S.2.</b> The Preprocessing Step outputs Intermediary Format with dataset split into training and validation/testing parts along with the Dataset Feature Specification yaml file. Metadata in the preprocessing step is automatically calculated.
* <b>S.3.</b> Intermediary Format data, together with the Dataset Feature Specification, are fed into training/evaluation scripts. The data loader reads Intermediary Format and feeds the data into the model according to the description in the Dataset Feature Specification.
* <b>S.4.</b> The model is trained and evaluated



<p align="center">
  <img src="./images/df_diagram.png" />
  <br>

Figure 3. Data flow in Recommender models in NVIDIA Deep Learning Examples repository. Channels of the model are drawn in green</a>.
</p>


#### Example of dataset feature specification

As an example, let’s consider a Dataset Feature Specification for a small CSV dataset for some abstract model.

```yaml
feature_spec:
  user_gender:
    dtype: torch.int8
    cardinality: 3 #M,F,Other
  user_age: #treated as numeric value
    dtype: torch.int8
  user_id:
    dtype: torch.int32
    cardinality: 2655
  item_id:
    dtype: torch.int32
    cardinality: 856
  label:
    dtype: torch.float32

source_spec:
  train:
    - type: csv
      features:
        - user_gender
        - user_age
      files:
        - train_data_0_0.csv
        - train_data_0_1.csv
    - type: csv
      features:
        - user_id
        - item_id
        - label
      files:
        - train_data_1.csv
  test:
    - type: csv
      features:
        - user_id
        - item_id
        - label
        - user_gender
        - user_age
        
      files:
        - test_data.csv

channel_spec:
  numeric_inputs: 
    - user_age
  categorical_user_inputs: 
    - user_gender
    - user_id
  categorical_item_inputs: 
    - item_id
  label_ch: 
    - label
```


The data contains five features: (user_gender, user_age, user_id, item_id, label). Their data types and necessary metadata are described in the feature specification section.

In the source mapping section, two mappings are provided: one describes the layout of the training data, and the other of the testing data. The layout for training data has been chosen arbitrarily to showcase the flexibility.
The train mapping consists of two chunks. The first one contains user_gender and user_age, saved as a CSV, and is further broken down into two files. For specifics of the layout, refer to the following example and consult the glossary. The second chunk contains the remaining columns and is saved in a single file. Notice that the order of columns is different in the second chunk - this is alright, as long as the order matches the order in that file (that is, columns in the .csv are also switched)


Let’s break down the train source mapping. The table contains example data color-paired to the files containing it.

<p align="center">
<img width="70%" src="./images/layout_example.png" />
</p>



The channel spec describes how the data will be consumed. Four streams will be produced and available to the script/model.
The feature specification does not specify what happens further: names of these streams are only lookup constants defined by the model/script.
Based on this example, we can speculate that the model has three  input channels: numeric_inputs, categorical_user_inputs,
categorical_item_inputs, and one  output channel: label.
Feature names are internal to the FeatureSpec and can be freely modified.


#### BYO dataset functionality

In order to train any Recommendation model in NVIDIA Deep Learning Examples, one can follow one of three possible ways:
* One delivers preprocessed datasets in the Intermediary Format supported by data loader used by the training script
(different models use different data loaders) together with FeatureSpec yaml file describing at least specification of dataset, features, and model channels

* One uses a transcoding script (**not supported in SIM model yet**)

* One delivers datasets in non-preprocessed form and uses preprocessing scripts that are a part of the model repository.
In order to use already existing preprocessing scripts, the format of the dataset needs to match one of the original datasets.
This way, the FeatureSpec file will be generated automatically, but the user will have the same preprocessing as in the original model repository. 

### Glossary

**Auxiliary loss** is used to improve DIEN (so SIM as well) model training. It is constructed based on consecutive user actions from their short behavior history.

**DIEN model** was proposed in [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) paper as an extension of the DIN model. It can also be used as a backbone for processing short interaction sequences in the SIM model.

**DIN model** was proposed in [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) paper. It can be used as a backbone for processing short interaction sequences in the SIM model.

**Long user behavior history** is the record of past user interactions. They are processed by the General Search Unit part of the SIM model (refer to Figure 1). This typically is a lightweight model aimed at processing longer sequences.

**Short user behavior history** is the record of the most recent user interactions. They are processed by a more computationally intensive Exact Search Unit part of the SIM model (refer to Figure 1).

**User behaviors** are users' interactions with given items of interest. Example interactions include reviewed items for Amazon Reviews dataset or clicks in the e-commerce domain. All the systems contained in this repository focus on modeling user interactions.

## Setup

The following section lists the requirements that you need to meet in order to start training the SIM model.

### Requirements

This repository contains a Dockerfile that extends the TensorFflow2 NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow2 22.01-py3](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) NGC container
- Supported GPUs:
  - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, refer to the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
  
For those unable to use the Tensorflow2 NGC container, to set up the required environment or create your own container, refer to the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the SIM model on the Amazon Reviews dataset. For the specifics concerning training and inference, refer to the [Advanced](#advanced) section.

1. Clone the repository.
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples
   cd DeepLearningExamples/TensorFlow2/Recommendation/SIM
   ```

2. Build the SIM Tensorflow2 container.
   ```bash
   docker build -t sim_tf2 .
   ```

3. Start an interactive session in the NGC container to run preprocessing, training, or inference (Amazon Books dataset can be mounted if it has already been downloaded, otherwise, refer to point 4). The SIM TensorFlow2 container can be launched with:
   ```bash
   docker run --runtime=nvidia -it --rm --ipc=host --security-opt seccomp=unconfined -v ${AMAZON_DATASET_PATH}:${RAW_DATASET_PATH} sim_tf2 bash
   ```

4. (Optional) Download Amazon Books dataset:

    ```bash
    scripts/download_amazon_books_2014.sh
    export RAW_DATASET_PATH=/data/amazon_books_2014
    ```


5. Start preprocessing.

    For details of the required file format and certain preprocessing parameters refer to [BYO dataset](#byo-dataset).

   ```bash
   python preprocessing/sim_preprocessing.py \
    --amazon_dataset_path ${RAW_DATASET_PATH} \
    --output_path ${PARQUET_PATH}

   python preprocessing/parquet_to_tfrecord.py \
    --amazon_dataset_path ${PARQUET_PATH} \
    --tfrecord_output_dir ${TF_RECORD_PATH}
   ```




For the explanation of output logs, refer to [Log format](#log-format) section.

Now that you have your model trained and evaluated, you can choose to compare your training results with our [Training accuracy results](#training-accuracy-results). You can also choose to benchmark your performance to [Training performance benchmark](#training-performance-results), or [Inference performance benchmark](#inference-performance-results). Following the steps in these sections will ensure that you achieve the same accuracy and performance results as stated in the [Results](#results) section.

## Advanced

The following sections provide greater details of the dataset, running training and inference, and the training results.

### Scripts and sample code

The `main.py` script provides an entry point to all the provided functionalities. This includes running training and inference modes. The behavior of the script is controlled by command-line arguments listed below in the [Parameters](#parameters) section. The `preprocessing` folder contains scripts to prepare data. In particular, the `preprocessing/sim_preprocessing.py` script can be used to preprocess the Amazon Reviews dataset while `preprocessing/parquet_to_tfrecord.py` transforms parquet files to TFRecords for loading data efficiently.

Models are implemented in corresponding modules in the `sim/models` subdirectory, for example, `sim/models/sim_model.py` for the SIM model. The `sim/layers` module contains definitions of different building blocks for the models. Finally, the `sim/data` subdirectory provides modules for the dataloader. Other useful utilities are contained in the `sim/utils` module.


### Getting the data

The SIM model was trained on the Books department subset of [Amazon Reviews](https://snap.stanford.edu/data/web-Amazon.html) dataset. The dataset is split into two parts: training and test data. The test set for evaluation was constructed using the last user interaction from user behavior sequences. All the preceding interactions are used for training.

This repository contains the `scripts/download_amazon_books_2014.sh`, which can be used to download the dataset.

#### Dataset guidelines

The preprocessing steps applied to the raw data include:
- Sampling negatives randomly (out of all possible items)
- Choosing the last category as the item category (in case more than one is available)
- Determining embedding table sizes for categorical features needed to construct a model
- Filter users for training split based on their number of interactions (discard users with less than 20 interactions)

#### Prebatching

Preprocessing scripts allow to apply batching prior to the model`s dataloader. This reduces the size of produced TFrecord files and speeds up dataloading.
To do so, specify `--prebatch_train_size` and `--prebatch_test_size` while converting data using `scripts/parquet_to_tfrecord.py`. Later, while using the `main.py` script, pass the information about applied prebatch size via the same parameters.

Example

Start preprocessing from step 5. from [Quick Start Guide](#quick-start-guide):

```bash
python preprocessing/sim_preprocessing.py \
--amazon_dataset_path ${RAW_DATASET_PATH} \
--output_path ${PARQUET_PATH}

python preprocessing/parquet_to_tfrecord.py \
--amazon_dataset_path ${PARQUET_PATH} \
--tfrecord_output_dir ${TF_RECORD_PATH} \
--prebatch_train_size ${PREBATCH_TRAIN_SIZE} \
--prebatch_train_size ${PREBATCH_TEST_SIZE}
```

And then train the model (step 6.):

```bash
mpiexec --allow-run-as-root --bind-to socket -np ${GPU} python main.py \
--dataset_dir ${TF_RECORD_PATH} \
--mode train \
--model_type sim \
--embedding_dim 16 \
--drop_remainder \
--optimizer adam \
--lr 0.01 \
--epochs 3 \
--global_batch_size 131072 \
--amp \
--prebatch_train_size ${PREBATCH_TRAIN_SIZE} \
--prebatch_train_size ${PREBATCH_TEST_SIZE}
```

<details>
<summary><b>Prebatching details</b></summary>

- The last batch for each split will pe saved to the separate file `remainder.tfrecord` unless there are enough samples to form a full batch.
- Final batch size used in main script can be a multiple of prebatch size.
- Final batch size used in main script can be a divider of prebatch size. In this case, when using multi GPU training, the number of batches received by each worker can be greater than 1 thus resulting in error during allgather operation. Dataset size, batch size and prebatch size have to be chosen with that limitation in mind.
- For the orignal Amazon Books Dataset, parameters were set to PREBATCH_TRAIN_SIZE = PREBATCH_TEST_SIZE = 4096 for performance benchmarking purposes.
</details>

&nbsp;

#### BYO dataset 

This implementation supports using other datasets thanks to BYO dataset functionality. 
BYO dataset functionality allows users to plug in their dataset in a common fashion for all Recommender models 
that support this functionality. Using BYO dataset functionality, the user does not have to modify the source code of 
the model thanks to the Feature Specification file. For general information on how the BYO dataset works, refer to the 
[BYO dataset overview section](#byo-dataset-functionality-overview).

For usage of preprocessing scripts refer to [Quick Start Guide](#quick-start-guide)

There are currently two ways to plug in the user's dataset:

<details>
<summary><b>1. Provide preprocessed dataset in parquet format, then use parquet_to_tfrecord.py script to convert it to Intermediary Format and automatically generate FeatureSpec.</b></summary>



Parquet $DATASET needs to have the following directory structure (or change the name with script arguments):

```
DATASET:
  metadata.json
  test:
    part.0.parquet
    part.1.parquet
    .
    .
    .
  train:
    part.0.parquet
    part.1.parquet
    .
    .
    .
```

`metadata.json` should contain cardinalities of each categorical feature present in the dataset and be of the following structure: (for features `uid`, `item`, `cat`)

```yaml
{
  "cardinalities": [
    {"name": "uid", "value": 105925}, 
    {"name": "item", "value": 1209081}, 
    {"name": "cat", "value": 2330}
  ]
}
```

Make sure the dataset's columns are in the same order as entries in `metadata.json` (for user features and **item features in each channel**)

Columns of parquets files must be organized in a specific order:
- one column with `label` values
- `number_of_user_features` (to be specified in script argument) columns follow with each **user feature** in the separate column
- `number_of_item_features` columns. One column for each feature of **target (query) item**
- `number_of_item_features` columns. Column with index i contains **sequence of item_feature_{i}** of **positive_history**
- `number_of_item_features` columns. Column with index i contains **sequence of item_feature_{i} **of **negative_history**

</details>

<details>
<summary><b>2. Provide preprocessed dataset in tfrecord format with feature_spec.yaml describing the details. </b></summary>

Required channels and sample layout can be found in the configuration shown below. This is the file layout and feature specification for the original Amazon dataset.

Files layout:
```
TF_RECORD_PATH:
    feature_spec.yaml
    test.tfrecord
    train.tfrecord
```

feature_spec.yaml:
```yaml
channel_spec:
  label:
  - label
  negative_history:
  - item_id_neg
  - cat_id_neg
  positive_history:
  - item_id_pos
  - cat_id_pos
  target_item_features:
  - item_id_trgt
  - cat_id_trgt
  user_features:
  - user_id
feature_spec:
  item_id_neg:
    cardinality: 1209081
    dimensions:
    - 100
    dtype: int64
  item_id_pos:
    cardinality: 1209081
    dimensions:
    - 100
    dtype: int64
  item_id_trgt:
    cardinality: 1209081
    dtype: int64
  cat_id_neg:
    cardinality: 2330
    dimensions:
    - 100
    dtype: int64
  cat_id_pos:
    cardinality: 2330
    dimensions:
    - 100
    dtype: int64
  cat_id_trgt:
    cardinality: 2330
    dtype: int64
  label:
    dtype: bool
  user_id:
    cardinality: 105925
    dtype: int64
metadata: {}
source_spec:
  test:
  - features: &id001
    - label
    - user_id
    - item_id_trgt
    - cat_id_trgt
    - item_id_pos
    - cat_id_pos
    - item_id_neg
    - cat_id_neg
    files:
    - test.tfrecord
    type: tfrecord
  train:
  - features: *id001
    files:
    - train.tfrecord
    type: tfrecord
```

`dimensions` should contain the length of the sequencial features.

Note that corresponsive features in `negative_history`, `positive_history`, `target_item_features` need to be listed in the same order in channel spec in each channel since they share embedding tables in the model. (for example `item_id` needs to be first and `cat_id` second). 

 </details>

 &nbsp;

##### Channel definitions and requirements

This model defines five channels:

- label, accepting a single feature
- negative_history, accepting a categorical ragged tensor for an arbitrary number of features
- positive_history, accepting a categorical ragged tensor for an arbitrary number of features
- target_item_features, accepting an arbitrary number of categorical features
- user_features, accepting an arbitrary number of categorical features

Features in `negative_history`, `positive_history` and `target_item_features` channels must be equal in number and must be defined in the same order in channel spec.

The training script expects two mappings:

- train
- test

For performance reasons, the only supported dataset type is tfrecord.

