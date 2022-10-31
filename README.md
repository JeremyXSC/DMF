[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-green.svg)
![GitHub stars](https://img.shields.io/github/stars/JeremyXSC/DMF.svg?style=flat&label=Star)

# DMF

## Deep Multimodal Fusion for Generalizable Person Reidentification

<img src='images/DMF.pdf'/>

This is the official implementation of our paper [Deep Multimodal Fusion for Generalizable Person Reidentification](). 

### News
- Support Market1501, CUHK03 and MSMT17 datasets.



### TODO
Write the documents

### Requirements
- torch
- torchvision
- timm
- yacs
- opencv-python
- fairseq

### How to use it?
This repo. supports training on multiple GPUs and the default setting is also multi-GPU.

1. Download all necessry datasets, e.g. DukeMTMC-reID, Market-1501 and CUHK03 datasets and move them to 'data'.  
2. Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```
3. Testing
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py
```

### Acknowledgments
This work was supported by the National Natural Science Foundation of China under Project(Grant No. 61977045 and Grant No. 81974276).
If you have further questions and suggestions, please feel free to contact us (xiangsuncheng17@sjtu.edu.cn).