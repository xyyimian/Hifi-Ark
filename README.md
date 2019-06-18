# Hifi-Ark
source code for paper # 2793: "Hi-Fi Ark: Deep User Representation via High-Fidelity Archive Network"

## Introduction
Thiis is an implementation of the paper [Hi-Fi Ark: Deep User Representation via High-Fidelity Archive Network](http://home.ustc.edu.cn/~xyyimian/IJCAI%202019%20submission.pdf) Zhengliu, Yu Xing, Fangzhao Wu, Mingxiao An, Xing Xie

Bibtex:




## Requirements
* Python >= 3.5
* Numpy >= 1.14.1 (Probably earlier version should work too)
* Keras >= 2.2.4
* Tensorflow >= 1.8.0
* GPU with memory >= 10G

## Training and Evaluation
Step into the root directory and run
```
python main.py models='model_name' rounds=1 epochs=5
```
Other arguments can be modified in the settings.py