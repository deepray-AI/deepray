## Quick Start Guide

To prepare the Criteo 1TB dataset for training, follow these steps.  

1. Make sure you meet the prerequisites.

You will need around 4TB of storage for storing the original Criteo 1TB dataset, the results of some
intermediate preprocessing steps and the final dataset. The final dataset itself will take about 400GB.

We recommend using local storage, such as a fast SSD drive, to run the preprocessing. Using other types of storage
will negatively impact the preprocessing time.


2. Build the preprocessing docker image.
```bash
docker build -t preproc_docker_image -f Dockerfile_spark . --build-arg DGX_VERSION=[DGX-2|DGX-A100]
```

3. Download the data by following the instructions at: http://labs.criteo.com/2013/12/download-terabyte-click-logs/.

When you have successfully downloaded the dataset, put it in the `/data/criteo_orig` directory in the container
(`$PWD/data/criteo_orig` in the host system).

4. Start an interactive session in the NGC container to run preprocessing.
The DLRM TensorFlow container can be launched with:

```bash
mkdir -p data
docker run --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data preproc_docker_image bash
```

5. Unzip the data with:

```bash
gunzip /data/criteo_orig/*.gz
```

6. Preprocess the data.

Here are a few examples of different preprocessing commands. Out of the box, we support preprocessing  on DGX-2 and DGX A100 systems. For the details on how those scripts work and detailed description of dataset types (small FL=15, large FL=3, xlarge FL=2), system requirements, setup instructions for different systems  and all the parameters consult the [preprocessing section](#preprocessing).
For an explanation of the `FL` parameter, see the [Dataset Guidelines](#dataset-guidelines) and [Preprocessing](#preprocessing) sections. 

Depending on dataset type (small FL=15, large FL=3, xlarge FL=2) run one of following command:

```bash
export download_dir=/data/criteo_orig
export final_output_dir=/data/preprocessed

cd preproc

# Preprocess to small dataset (FL=15) with Spark GPU:
./prepare_dataset.sh 15 GPU Spark

# Preprocess to large dataset (FL=3) with Spark GPU:
./prepare_dataset.sh 3 GPU Spark

# Preprocess to xlarge dataset (FL=2) with Spark GPU:
./prepare_dataset.sh 2 GPU Spark

# to run on Spark GPU with no frequency limit:
./prepare_dataset.sh 0 GPU Spark
```



## Advanced

### Dataset guidelines

The first 23 days are used as the training set. The last day is split in half.
The first part is used as a validation set and the second set is used as a hold-out test set.

The preprocessing steps applied to the raw data include:
- Replacing the missing values with `0`.
- Replacing the categorical values that exist fewer than 15 times with a special value.
- Converting the hash values to consecutive integers.
- Adding 2 to all the numerical features so that all of them are greater or equal to 1.
- Taking a natural logarithm of all numerical features.


### Preprocess with Spark

The preprocessing scripts provided in this repository support running both on CPU and GPU using [NVtabular](https://developer.nvidia.com/blog/announcing-the-nvtabular-open-beta-with-multi-gpu-support-and-new-data-loaders/) (GPU only) and [Apache Spark 3.0](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/apache-spark-3/).

Please note that the preprocessing will require about 4TB of disk storage. 


The syntax for the preprocessing script is as follows:
```bash
cd /workspace/dlrm/preproc
./prepare_dataset.sh <frequency_threshold> <GPU|CPU> <NVTabular|Spark>
```

For the Criteo Terabyte dataset, we recommend a frequency threshold of `FL=3`(when using A100 40GB or V100 32 GB) or `FL=2`(when using A100 80GB) if you intend to run the hybrid-parallel mode
on multiple GPUs. If you want to make the model fit into a single NVIDIA Tesla V100-32GB, you can set `FL=15`. 

The first argument means the frequency threshold to apply to the categorical variables. For a frequency threshold `FL`, the categorical values that occur less 
often than `FL` will be replaced with one special value for each category. Thus, a larger value of `FL` will require smaller embedding tables 
and will substantially reduce the overall size of the model.

The second argument is the hardware to use (either GPU or CPU).  

The third arguments is a framework to use (either NVTabular or Spark). In case of choosing a CPU preprocessing this argument is omitted as it only Apache Spark is supported on CPU.

The preprocessing scripts make use of the following environment variables to configure the data directory paths:
- `download_dir` – this directory should contain the original Criteo Terabyte CSV files
- `spark_output_path` – directory to which the parquet data will be written
- `conversion_intermediate_dir` – directory used for storing intermediate data used to convert from parquet to train-ready format
- `final_output_dir` – directory to store the final results of the preprocessing which can then be used to train DLRM 

In the `final_output_dir` will be three subdirectories created: `train`, `test`, `validation`, and one json file &ndash; `model_size.json` &ndash; containing a maximal index of each category. 
The `train` is the train dataset transformed from day_0 to day_22. 
The `test` is the test dataset transformed from the prior half of day_23. 
The `validation` is the dataset transformed from the latter half of day_23.

The model is tested on 3 datasets resulting from Criteo dataset preprocessing: small (Freqency threshold = 15), large (Freqency threshold = 3) and xlarge (Freqency threshold = 2). Each dataset occupies approx 370GB of disk space. Table below presents information on the supercomputer and GPU count that are needed to train model on particular dataset.

| Dataset | GPU VRAM consumption\* | Model checkpoint size\* | FL setting | DGX A100 40GB, 1GPU | DGX A100 40GB, 8GPU | DGX A100 80GB, 1GPU | DGX A100 80GB, 8GPU | DGX-1** or DGX-2, 1 GPU | DGX-1** or DGX-2, 8GPU | DGX-2, 16GPU |
| ------- | ---------------------- | ----------------------- | ---------- | -------------------- | -------------------- | -------------------- | -------------------- | ---------------------- | --------------------- | ------------ |
| small (FL=15) | 20.5 | 15.0 | 15 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| large (FL=3) | 132.3 | 81.9 | 3 | NA | Yes | NA | Yes | NA | Yes | Yes |
| xlarge (FL=2) | 198.8 | 141.3 | 2 | NA | NA | NA | Yes | NA | NA | NA |

\*with default embedding dimension setting
\**DGX-1 V100 32GB

##### NVTabular

NVTabular preprocessing is calibrated to run on [DGX A100](https://www.nvidia.com/en-us/data-center/dgx-a100/) and [DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2/) AI systems. However, it should be possible to change the values of `ALL_DS_MEM_FRAC`, `TRAIN_DS_MEM_FRAC`, `TEST_DS_MEM_FRAC`, `VALID_DS_MEM_FRAC` in `preproc/preproc_NVTabular.py`, so that they'll work on also on other hardware platforms such as DGX-1 or a custom one. 

##### Spark

The script `spark_data_utils.py` is a PySpark application, which is used to preprocess the Criteo Terabyte Dataset. In the Docker image, we have installed Spark 3.0.1, which will start a standalone cluster of Spark. The scripts `run_spark_cpu.sh` and `run_spark_gpu.sh` start Spark, then run several PySpark jobs with `spark_data_utils.py`. 

Note that the Spark job requires about 3TB disk space used for data shuffling.

Spark preprocessing is calibrated to run on [DGX A100](https://www.nvidia.com/en-us/data-center/dgx-a100/) and [DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2/) AI systems. However, it should be possible to change the values in `preproc/DGX-2_config.sh` or `preproc/DGX-A100_config.sh`
so that they'll work on also on other hardware platforms such as DGX-1 or a custom one. 





The preprocessing scripts makes use of the following environment variables to configure the data directory paths:
- `download_dir` – this directory should contain the original Criteo Terabyte CSV files
- `spark_output_path` – directory to which the parquet data will be written
- `conversion_intermediate_dir` – directory used for storing intermediate data used to convert from parquet to train-ready format
- `final_output_dir` – directory to store the final results of the preprocessing which can then be used to train DLRM

The script `spark_data_utils.py` is a PySpark application, which is used to preprocess the Criteo Terabyte Dataset. In the Docker image, we have installed Spark 3.0.1, which will start a standalone cluster of Spark. The scripts `run_spark_cpu.sh` and `run_spark_gpu.sh` start Spark, then runs several PySpark jobs with `spark_data_utils.py`, for example:
generates the dictionary
- transforms the train dataset
- transforms the test dataset
- transforms the validation dataset

    Change the variables in the `run-spark.sh` script according to your environment.
    Configure the paths.
```
export SPARK_LOCAL_DIRS=/data/spark-tmp
export INPUT_PATH=/data/criteo
export OUTPUT_PATH=/data/output
```
Note that the Spark job requires about 3TB disk space used for data shuffle.

Where:
`SPARK_LOCAL_DIRS` is the path where Spark uses to write shuffle data.
`INPUT_PATH` is the path of the Criteo Terabyte Dataset, including uncompressed files like day_0, day_1…
`OUTPUT_PATH` is where the script writes the output data. It will generate the following subdirectories of `models`, `train`, `test`, and `validation`.
- The `model` is the dictionary folder.
- The `train` is the train dataset transformed from day_0 to day_22.
- The `test` is the test dataset transformed from the prior half of day_23.
- The `validation` is the dataset transformed from the latter half of day_23.

Configure the resources which Spark will use.
```
export TOTAL_CORES=80
export TOTAL_MEMORY=800
```

Where:
`TOTAL_CORES` is the total CPU cores you want Spark to use.

`TOTAL_MEMORY` is the total memory Spark will use.

Configure frequency limit.
```
USE_FREQUENCY_LIMIT=15
```
The frequency limit is used to filter out the categorical values which appear less than n times in the whole dataset, and make them be 0. Change this variable to 1 to enable it. The default frequency limit is 15 in the script. You also can change the number as you want by changing the line of `OPTS="--frequency_limit 8"`.

