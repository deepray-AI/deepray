Jarvis is a toolbox built on top of TensorFlow2.0 that allows developers and researchers to easily build neural networks in TensorFlow, particularly CTR models for large-scale advertising and recommendation scenarios. It provides the implementation of [Meitu's FLEN model](https://arxiv.org/abs/1911.04690).

Note that Jarvis is **still actively under development**, so feedback and contributions are welcome.
Feel free to submit your contributions as a pull request.

Jarvis features:

- Scalability: fast training on large-scale networks with tens of millions of sparse features
- Extensible: easily register new models and criteria.
- Supported tasks:
  - CTR prediction
  - Multi-task learning (coming)
  - online learning (todo)

## Getting Started

## Requirements and Installation
Please see environment.yml for more details

## Usage

You can use `python scripts/flen.py` to run **FLEN** model on Avazu dataset.

Expected output:

| Variant  | AUC    | Logloss |
|----------|--------|---------|
| FLEN     | 0.7519 | 0.3944  | 
| FLEND    | 0.7528 | 0.3944  | 

### Avazu dataset
Download the tfrecord format dataset from [here](https://www.dropbox.com/s/unabeqg8fm0ezxx/tiny_groups.tar.gz?dl=0).  
Alternatively, You can use `python tools/dataset/avazu.py` to prepare Avazu dataset yourself. 


## Customization

### Implement Your Own Model

If you have a well-perform algorithm and are willing to implement it in our toolkit to help more people, you can create a pull request,  detailed information can be found [here](https://help.github.com/en/articles/creating-a-pull-request). 
