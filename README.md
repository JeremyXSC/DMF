[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-green.svg)
![GitHub stars](https://img.shields.io/github/stars/JeremyXSC/DMF.svg?style=flat&label=Star)

# DMF

## Deep Multimodal Fusion for Generalizable Person Reidentification

<img src='images/DMF.png'/>

This is the official implementation of our paper [Deep Multimodal Fusion for Generalizable Person Re-identification](https://arxiv.org/pdf/2211.00933.pdf). And the pretrained models can be downloaded from [data2vec](https://github.com/facebookresearch/data2vec_vision/tree/main/beit).

### News
- Support Market1501, CUHK03, MSMT17 and RandPerson datasets.




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
This repo. supports training on multiple GPUs and the default setting is single GPU (One RTX 3090 GPU).

1. Download all necessry datasets (e.g. Market1501, CUHK03 and MSMT17 datasets) and move them to 'data'.   
2. Training
```
python train.py
```
3. Testing
```
python test.py
```

### Experiment Results on Market-1501, CUHK03 and MSMT17 datasets.
<!-- markdownlint-disable MD033 -->
<table>
    <tr>
        <th rowspan="2" align="center">Dataset for fine-tuning</th>
        <th colspan="2" align="center">Market-1501</th>
        <th colspan="2" align="center">CUHK03</th>
		<th colspan="2" align="center">MSMT17</th>
        <th rowspan="2" align="center">Settings</th>
    </tr>
    <tr>
        <td align="center">Rank-1</td>
        <td align="center">mAP</td>
		<td align="center">Rank-1</td>
        <td align="center">mAP</td>
        <td align="center">Rank-1</td>
        <td align="center">mAP</td>
    </tr>
    <tr><td>Market-1501</td><td align="center">--</td><td align="center">--</td><td align="center">23.4</td><td align="center">22.6</td><td align="center">50.6</td><td align="center">21.5</td><td align="center">1GPU</td></tr>
    <tr><td>MSMT17</td><td align="center">81.3</td><td align="center">55.1</td><td align="center">26.1</td><td align="center">24.7</td><td align="center">--</td><td align="center">--</td><td align="center">1GPU</td></tr>
    <tr><td>MSMT17<sub>all</sub></td><td align="center">82.6</td><td align="center">58.8</td><td align="center">34.0</td><td align="center">32.1</td><td align="center">--</td><td align="center">--</td><td align="center">1GPU</td></tr>
    <tr><td>RandPerson</td><td align="center">78.7</td><td align="center">52.0</td><td align="center">21.5</td><td align="center">19.3</td><td align="center">52.4</td><td align="center">18.9</td><td align="center">1GPU</td></tr>
</table>

### Acknowledgments
This work was supported by the National Natural Science Foundation of China under Projects (Grant No. 61977045 and Grant No. 81974276).
If you have further questions and suggestions, please feel free to contact us (xiangsuncheng17@sjtu.edu.cn).

If you find this code useful in your research, please consider citing:
```
@article{xiang2022deep,
  title={Deep Multimodal Fusion for Generalizable Person Re-identification},
  author={Xiang, Suncheng and Chen, Hao and Gao, Jingsheng and Du, Sijia and Mou, Jiawang and Liu, Ting and Qian, Dahong and Fu, Yuzhuo},
  journal={arXiv preprint arXiv:2211.00933},
  year={2022},
}
```