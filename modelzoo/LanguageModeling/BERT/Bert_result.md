| Parameters            | A100                    | A800      |
| --------------------- | ----------------------- | ----------------------- |
| DataSet               | SQuAD1.1                | SQuAD1.1              |
| num_hidden_layers     | 12                      | 12                      |
| batch_size_per_gpu    | 32                   | 32                    |
| learning_rate_per_gpu | 5e-6                    | 5e-6                    |
| precision             | fp16                    | Fp16                  |
| use_xla               | true                | true           |
| num_gpus              | 8                       | 8                       |
| max_seq_length        | 384                     | 384                     |
| doc_stride            | 128                     | 128                     |
| epochs                | 1                       | 1                     |
| checkpoint            | uncased_L-12_H-768_A-12 | uncased_L-12_H-768_A-12 |

| Task | total_training_steps | train_loss | F1         | exact_match | Throughput Average (sentences/sec)! | Training Duration sec | GPU Util | GPU Memory-Usage(MB)! |
| -------------- | ---------- | ----------------------------------- | --------------------- | -------- | --------------------- | --------------------- | --------------------- | --------------------- |
| A800_GPU-8_bs-12_LR-5e-6_fp16_XLA-true_BERT-base_SQuAD1.1_Epoch-1 |  |  | 85.2391 | 76.6982 | 280.94 | 1011.48 | 8 * 96% |  |
|  |                      |                    |         |             |                                     |                             |          |                       |
|                                                              |                      |                    |         |             |                                     |                             |          |                       |
|                                                              |                      |                    |         |             |                                     |                             |          |                       |
|                                                              |                      |                    |         |             |                                     |                             |          |             |
| TITAN_GPU-4_bs-12_LR-5e-6_fp16_XLA-true_BERT-base_SQuAD1.1_Epoch-1 | 1846 | 1.0677543878555298 | 84.4242 | 75.4494 | 286.32 | 511.30 for Examples = 88608 | 8 * 96% |  |

*Memory(GiB): Max consumption

*CPU Util: Max moment value
