# [LocalTrans: A Multiscale Local Transformer Network for Cross-Resolution Homography Estimation](http://www.liuyebin.com/localtrans/localtrans.html)
Ruizhi Shao*, Gaochang Wu*, Yuemei Zhou, Ying Fu, Lu Fang, Yebin Liu

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2106.04067)

This repository contains the official pytorch implementation of ”*LocalTrans: A Multiscale Local Transformer Network for Cross-Resolution Homography Estimation*“.

![Teaser Image](assets/teaser.jpg)

## Requirements
- pytorch
- matplotlib
- numpy
- cv2
- tensorboard
- kornia
- imageio

## Training
To train localtrans on the COCO dataset in different setting, run the following code:
```
sh train.sh
```

## Testing
Run the following code to test on the COCO test dataset.
```
sh test.sh
```

## Citation
```
@inproceedings{shao2021localtrans,
title={LocalTrans: A Multiscale Local Transformer Network for Cross-Resolution Homography Estimation},
author={Shao, Ruizhi and Wu, Gaochang and Zhou, Yuemei and Fu, Ying and Fang, Lu and Liu, Yebin},
booktitle={IEEE Conference on Computer Vision (ICCV 2021)},
year={2021},
}
```
