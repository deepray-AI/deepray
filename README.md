## **Introduction**
Deepray is a deep learning framework for Keras, to build model like LEGO, and train model with easier, faster and cheaper way.


## **Why Deepray?**
Deepray contains list of features to improve usability and performance for Deep Learning, especially provides some essential components for recommendation algorithm. 


**Trainer**
 - Distributed Training with Horovod backend
 - Gradient accumulation

**Layers**
 - Embedding Variable from [DeepRec](https://github.com/DeepRec-AI/DeepRec).
 - Compositional Embedding
 - Feature Cross layer for recommendation algorithm
 - ......

**Kernels**
 - Group Embedding for Embedding Variable
 - ......

**Optimizer**
 - Adam/Adagrad/SDG/FTRL Optimizer for Embedding Variable
 - AdamAsync Optimizer
 - MultiOptimizer
 - ......

**Datasets**
 - Parquet Dataset from [HybridBackend](https://github.com/DeepRec-AI/HybridBackend)
 - ......

**......**

#### Compatibility Matrix
|  Deepray        | TensorFlow |  Compiler  | cuDNN | CUDA |
| :-------------- | :--------- | :--------- | :---- | :--- |
| deepray-0.21.86 | 2.15       | GCC 11.4.0 | 8.9   | 12.3.2 |


# Quick start
 - Install Deepray:

```bash
pip install deepray
```

 - Using Docker(**Recommended**):
Latest Release Images: **hailinfufu/deepray-release:nightly-py3.10-tf2.15.0-cu12.3.2-ubuntu22.04**
```
docker pull hailinfufu/deepray-release:nightly-py3.10-tf2.15.0-cu12.3.2-ubuntu22.04
docker run -it hailinfufu/deepray-release:nightly-py3.10-tf2.15.0-cu12.3.2-ubuntu22.04
```

 - Build from source:
```
git clone https://github.com/deepray-AI/deepray.git
cd deepray && bash build.sh
```


### Deepray example
Define the training workflow. Here's a toy example ([explore real examples](https://github.com/deepray-AI/deepray/blob/main/modelzoo/Recommendation/CreditCardFraudDetection/train.py)):

```python
# main.py
# ! pip install deepray
from typing import Dict

import tensorflow as tf
from absl import flags

import deepray as dp
from deepray.core.trainer import Trainer
from deepray.datasets.movielens.movielens_100k_ratings import Movielens100kRating
from deepray.layers.embedding_variable import EmbeddingVariable

# --------------------------------
# Step 1: Define a Keras Module
# --------------------------------
class RankingModel(tf.keras.Model):

  def __init__(self, embedding_dimension=32):
    super().__init__()
    # Compute embeddings for users.
    self.user_embeddings = EmbeddingVariable(embedding_dim=embedding_dimension)
    self.movie_embeddings = EmbeddingVariable(embedding_dim=embedding_dimension)

    # Compute predictions.
    self.ratings = tf.keras.Sequential(
        [
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ]
    )

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    user_id, movie_title = inputs["user_id"], inputs["movie_title"]
    user_id = tf.reshape(user_id, [-1])
    movie_title = tf.reshape(movie_title, [-1])
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)
    emb_vec = tf.concat([user_embedding, movie_embedding], axis=1)
    return self.ratings(emb_vec)


# -------------------
# Step 2: Define data
# -------------------
data_pipe = Movielens100kRating(split=True)
dataset = data_pipe(flags.FLAGS.batch_size, is_training=True)

# -------------------
# Step 3: Train
# -------------------
optimizer = dp.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate, amsgrad=False)
model = RankingModel()
trainer = Trainer(model=model, optimizer=optimizer, loss="MSE", metrics=[tf.keras.metrics.RootMeanSquaredError()])
trainer.fit(x=dataset)
```

Run the model on your terminal

```bash
python main.py --batch_size=32 --learning_rate=0.03
```
----
## Examples

###### Recommender Systems

- [Deep & Cross Network V2 with Criteo](https://github.com/deepray-AI/deepray/tree/main/modelzoo/Recommendation/criteo_ctr)
- [MovieLens](https://github.com/deepray-AI/deepray/tree/main/modelzoo/Recommendation)

###### Natural Language Processing

- [BERT](https://github.com/deepray-AI/deepray/tree/main/modelzoo/LanguageModeling/BERT)

###### Computer Vision

- [Mnist](https://github.com/deepray-AI/deepray/tree/main/modelzoo/CV/mnist)

## Communication

- [GitHub issues](https://github.com/deepray-AI/deepray/issues): any install, bug, feature issues.
- 微信号: StateOfArt

## License
[Apache License 2.0](LICENSE)

