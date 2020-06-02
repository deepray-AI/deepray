**DeePray** (`深度祈祷`): A new Modular, Scalable, Configurable, Easy-to-Use and Extend infrastructure for Deep Learning based Recommendation.

Let's Get Started!

```
pip install deepray
```

https://deepray.readthedocs.io/en/latest/

Models List

| Titile                                                       |  Booktitle  | Resources                                                    |
| ------------------------------------------------------------ | :---------: | ------------------------------------------------------------ |
| **FM**: Factorization Machines                               |  ICDM'2010  | [[pdf]](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_fm.py) |
| **FFM**: Field-aware Factorization Machines for CTR Prediction | RecSys'2016 | [[pdf]](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_ffm.py) |
| **FNN**: Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction |  ECIR'2016  | [[pdf]](https://arxiv.org/abs/1601.02376)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_fnn.py) |
| **PNN**: Product-based Neural Networks for User Response Prediction |  ICDM'2016  | [[pdf]](https://arxiv.org/abs/1611.00144)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_pnn.py) |
| **Wide&Deep**: Wide & Deep Learning for Recommender Systems  |  DLRS'2016  | [[pdf]](https://arxiv.org/pdf/1606.07792)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_wdl.py) |
| **AFM**: Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks | IJCAI'2017  | [[pdf]](https://arxiv.org/abs/1708.04617)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_afm.py) |
| **NFM**: Neural Factorization Machines for Sparse Predictive Analytics | SIGIR'2017  | [[pdf]](https://arxiv.org/abs/1708.05027)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_nfm.py) |
| **DeepFM**: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[C] | IJCAI'2017  | [[pdf]](https://arxiv.org/abs/1703.04247) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_deepfm.py) |
| **DCN**: Deep & Cross Network for Ad Click Predictions       | ADKDD'2017  | [[pdf]](https://arxiv.org/abs/1708.05123) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dcn.py) |
| **xDeepFM**: xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems |  KDD'2018   | [[pdf]](https://arxiv.org/abs/1803.05170) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_xdeepfm.py) |
| **DIN**: DIN: Deep Interest Network for Click-Through Rate Prediction |  KDD'2018   | [[pdf]](https://arxiv.org/abs/1706.06978) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dien.py) |
| **DIEN**: DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction |  AAAI'2019  | [[pdf]](https://arxiv.org/abs/1809.03672) [[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dien.py) |
| **DSIN**: Deep Session Interest Network for Click-Through Rate Prediction | IJCAI'2019  | [[pdf]](https://arxiv.org/abs/1905.06482)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_dsin.py) |
| **AutoInt**: Automatic Feature Interaction Learning via Self-Attentive Neural Networks |  CIKM'2019  | [[pdf]](https://arxiv.org/abs/1810.11921)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_autoint.py) |
| **FLEN**: Leveraging Field for Scalable CTR Prediction       |  AAAI'2020  | [[pdf]](https://arxiv.org/pdf/1911.04690.pdf)[[code]](https://github.com/fuhailin/DeePray/blob/master/deepray/model/model_flen.py) |
| **DFN**: Deep Feedback Network for Recommendation            | IJCAI'2020  | [[pdf]]()[[code]](TODO)                                      |

# How to build your own model with DeePray

Inheriting   `BaseCTRModel` class from `from deepray.model.model_ctr`, and implement your own `build_network()` method!


# Contribution

DeePray is still under development, and call for contributions!

```
* Hailin Fu (`Hailin <https://github.com/fuhailin>`)
* Call for contributions!
```

# Citing
DeePray is designed, developed and supported by [Hailin](https://github.com/fuhailin/).
If you use any part of this library in your research, please cite it using the following BibTex entry
```latex
@misc{DeePray,
  author = {Hailin Fu},
  title = {DeePray: A new Modular, Scalable, Configurable, Easy-to-Use and Extend infrastructure for Deep Learning based Recommendation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/fuhailin/deepray}},
}
```

# License

Copyright (c) Copyright © 2020 The DeePray Authors<Hailin Fu>. All Rights Reserved.

Licensed under the [Apach](LICENSE) License.

# Contact
If you have any questions, please follow the following account:

<img src="https://gitee.com/fuhailin/Object-Storage-Service/raw/master/wechat_channel.png" >