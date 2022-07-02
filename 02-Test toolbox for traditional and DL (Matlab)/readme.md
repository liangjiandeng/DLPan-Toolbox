# Test toolbox for traditional and DL
"Test toolbox for traditional and DL" for simultaneously evaluating traditional and DL approaches, and finally output metrics and eps-format figures for your latex editing

[English](https://github.com/.md) | [简体中文](https://github.com.md)

This repository is the official Matlab implementation of our IEEE GRSM paper “Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks”, 2022 ([paper](https://liangjiandeng.github.io), [homepage](https://liangjiandeng.github.io)).



## Features


## Requirements
* Matlab software

## Quick Start

# Full-resolution Evaluation.** 

* Directly run ``Demo_Full_Resolution.m`` which includes an WV3 example. After running this demo, readers can understand the whole procedure.

* Note: the test dataset of full-resolution are too huge to upload to GitHub, thus we provide cloud links to readers to download them to
  successfully run this demo, including:
  - i) Download link for full-resolution WV3-NewYork example (named "NY1_WV3_FR.mat"): http:********   (put into the folder of   "1_TestData/Datasets Testing")
  
  - ii) Download link of DL's results for full-resolution WV3-NewYork example: http:********   (put into the folder of "'2_DL_Result/WV3")
  
* Once you have above datasets, you can run this demo successfully, then understand how this demo run!

**Reduced-resolution Evaluation.** 

* Directly run ``Demo_Reduced_Resolution.m`` which includes an WV3 example. After running this demo, readers can understand the whole procedure.


**Others**
* if you want to add customized datasets, you need to update:

```
01-DL toolbox (Pytorch)/UDL/AutoDL/__init__.py.
01-DL toolbox (Pytorch)/UDL/pansharpening/common/psdata.py.
```



## Contribution
We appreciate all contributions to improving '01-DL toolbox (Pytorch)'. Looking forward to your contribution to DLPan-Toolbox.


## Citation
Please cite this project if you use datasets or the toolbox in your research.
> 


## Acknowledgement
- We appreciate the great contribution of [Xiao Wu](https://xiaoxiao-woo.github.io/) who is a graduate student in [UESTC](https://www.uestc.edu.cn/) to this toolbox.

## License & Copyright
This project is open sourced under GNU General Public License v3.0.

