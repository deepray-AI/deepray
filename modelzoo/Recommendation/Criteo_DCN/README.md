## Model Structure
[Deep & Cross Network V2](https://arxiv.org/abs/2008.13535)(DCN V2) is proposed by Google by 2020.   

## Usage
1.  Please prepare the Deepray env.
    1. Manually
       - Download code by `git clone https://github.com/deepray-AI/deepray.git`
       - Follow [How to Build](https://github.com/deepray-AI/deepray?tab=readme-ov-file#installing-from-source) to build Deepray whl package and install by `pip install $DEEPRAY_WHL`.
       
    2. Docker
    
       ```
       docker pull hailinfufu/deepray-release:nightly-py3.10-tf2.15.0-cu12.2.2-ubuntu22.04
       docker run -it hailinfufu/deepray-release:nightly-py3.10-tf2.15.0-cu12.2.2-ubuntu22.04
       # In docker container
       cd /deepray/modelzoo/Recommendation/Criteo_ctr
       ```

    
2.  Stand-alone Training.  
    ```sh
    bash run.sh
    ```
    Distribute Training with 4 GPU cards.
    ```sh
    bash run.sh 4
    ```

## Dataset
Train & eval dataset using ***Kaggle Display Advertising Challenge Dataset (Criteo Dataset)***.
Here provided sampled two parquet files: 
https://drive.google.com/drive/folders/1VXYvru_KU5Lv0voWhUIkaioLznnudHJl?usp=sharing