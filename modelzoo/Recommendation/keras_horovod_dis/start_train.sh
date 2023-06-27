# gpu_num = nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
# horovodrun -np $gpu_num python keras_horovod_distributed_demo.py

horovodrun -np 4 python -m examples.Recommendation.keras_horovod_dis.keras_horovod_distributed_demo
