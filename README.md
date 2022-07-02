# DLPan-Toolbox

* This is a deep learning (DL) toolbox for pansharpening, which can be used for your training anda testing, and easily get the comparison of traditional and DL methdos.

* You may find the relate [paper](https://liangjiandeng.github.io/) which will be published in IEEE Geoscience and Remote Sensing Magazine, 2022.




## Introduction
This toolbox mainly contains two parts: one is the pytorch source codes for the eight DL-based methods in the paper (i.e., the folder of "01-DL toolbox (Pytorch)"); the other is the Matlab source codes which could simultaneously evaluate the performance of traditional and DL approaches in a uniformed framework ("02-Test toolbox for traditional and DL (Matlab)"). Please see more details:

- 01-DL toolbox (Pytorch): contains source codes of DL methods, you may check the ``readme`` file for the usage.
- 02-Test toolbox for traditional and DL (Matlab): contains Matlab source codes for simultaneously evaluating traditional and DL approaches and outputing results, you may check the ``readme`` file for the usage.

Note that, readers also could check the structure and relationship of these two folders in the following ``overview`` figure (also see it in the respository).


<img src="overview.png" width = "90%" />



## Citation
```bibtex
@ARTICLE{deng2022grsm,
author={L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza},
booktitle={IEEE Geoscience and Remote Sensing Magazine},
title={Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks},
year={2022},
pages={},
}
```
