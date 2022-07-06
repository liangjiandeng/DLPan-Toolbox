# Test toolbox for traditional and DL
"Test toolbox for traditional and DL" for simultaneously evaluating traditional and DL approaches, and finally output metrics and eps-format figures for your latex editing

[English](https://github.com/.md) | [简体中文](https://github.com.md)


This repository is the official Matlab implementation of our IEEE GRSM paper “Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks”, 2022 ([paper](https://github.com/liangjiandeng/liangjiandeng.github.io/tree/master/papers/2022/review-grsm2022.pdf) | [homepage](https://github.com/liangjiandeng/DLPan-Toolbox)).

## Features


## Requirements
* Matlab software

## Quick Start

### Full-resolution Evaluation

* Directly run ``Demo_Full_Resolution.m`` which includes an WV3 example. After running this demo, readers can understand the whole procedure.

* Note: the test dataset of full-resolution are too huge to upload to GitHub, thus we provide cloud links to readers to download them to
  successfully run this demo, including:
  - i) Download link for full-resolution WV3-NewYork example (named "NY1_WV3_FR.mat"): [[Link]](https://drive.google.com/file/d/1j1nyHuBxsNzIn-UEwZUgeziGCAFMLes9/view?usp=sharing)   (put into the folder of   "1_TestData/Datasets Testing")
  
  - ii) Download link of DL's results for full-resolution WV3-NewYork example: [[Link]](https://drive.google.com/file/d/16FSxdq6BY7STbmMzxcxJ5atNQ7ZV3mPT/view?usp=sharing)   (put into the folder of "'2_DL_Result/WV3")
  
* Once you have above datasets, you can run this demo successfully, then understand how this demo run!




### Reduced-resolution Evaluation

* Directly run ``Demo_Reduced_Resolution.m`` which includes an WV3 example. After running this demo, readers can understand the whole procedure.

* Note: the test dataset of reduced-resolution are too huge to upload to GitHub, thus we provide cloud links to readers to download them to
  successfully run this demo, including:
  - i) Download link for reduced-resolution WV3-NewYork example (named "NY1_WV3_RR.mat"): same link as above i), then put into the folder of   "1_TestData/Datasets Testing"
  
  - ii) Download link of DL's results for reduced-resolution WV3-NewYork example: same link as above ii), then put into the folder of "2_DL_Result/WV3"
  
* Once you have above datasets, you can run this demo successfully, then understand how this demo run!


### Others

* You may find the quantitative results from Tex files such as ``FR_Assessment.tex``, ``RR_Assessment.tex`` and ``Avg_RR_Assessment.tex``, then copy for your Latex editing.
* You may also find the generated high-resolution eps-format figures in the folder of "3_EPS" for your Latex editing. 


## Acknowledgement
- We appreciate the great contribution of [Xiao Wu](https://xiaoxiao-woo.github.io/) who is a graduate student in [UESTC](https://www.uestc.edu.cn/) to this toolbox.


## Citation
* If you use this toolbox, please kindly cite our paper:

```bibtex
@ARTICLE{deng2022grsm,
author={L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza},
booktitle={IEEE Geoscience and Remote Sensing Magazine},
title={Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks},
year={2022},
pages={},
}
```

* Also, the codes of traditional methods are from the "pansharpening toolbox for distribution", thus please cite the corresponding paper:
```bibtex
@ARTICLE{vivone2021grsm,
  author={Vivone, Gemine and Dalla Mura, Mauro and Garzelli, Andrea and Restaino, Rocco and Scarpa, Giuseppe and Ulfarsson, Magnus O. and   Alparone, Luciano and Chanussot, Jocelyn},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods}, 
  year={2021},
  volume={9},
  number={1},
  pages={53-81},
  doi={10.1109/MGRS.2020.3019315}
}
```

  


## License & Copyright
This project is open sourced under GNU General Public License v3.0.

