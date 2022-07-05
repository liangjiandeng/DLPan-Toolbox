# DL toolbox
"DL toolbox" for Remote Sensing Pansharpening

[English]([https://github.com/XiaoXiao-Woo/PanCollection/edit/dev/README.md](https://github.com/liangjiandeng/DLPan-Toolbox/edit/main/01-DL-toolbox(Pytorch)/readme.md)) | [简体中文](https://github.com.md)

This repository is the official PyTorch implementation of our IEEE GRSM paper “Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks”, 2022 ([paper](https://github.com/liangjiandeng/liangjiandeng.github.io/tree/master/papers/2022/review-grsm2022.pdf) | [homepage](https://github.com/liangjiandeng/DLPan-Toolbox)).



## Features


## Requirements
* Python3.7+, Pytorch>=1.6.0
* NVIDIA GPU + CUDA
* Run `python setup.py develop`

Note: Our project is based on MMCV, but you needn't to install it currently.

## Quick Start
**Step0.** Set your Python environment.

>git clone https://github.com/liangjiandeng/DLPan-Toolbox/tree/main/01-DL-toolbox(Pytorch)

Then, 

> python setup.py develop

**Step1.** Put datasets and set path
* Put datasets (WorldView-3, QuickBird, GaoFen2, WorldView2) into the `UDL/Data/pansharpening`, see following path structure. 

```
|-$ROOT/Data
├── pansharpening
│   ├── training_data
│   │   ├── train_wv3.h5
│   │   ├── ...
│   ├── validation_data
│   │   │   ├── valid_wv3.h5
│   │   │   ├── ...
│   ├── test_data
│   │   ├── WV3
│   │   │   ├── NY1_WV3_RR.mat
│   │   │   ├── ...
│   │   │   ├── ...
```

* Check and revise your dataset path in `01-DL-toolbox(Pytorch)/UDL/Basis/option.py` (may not need to revise), or you can print the output of `run_pansharpening.py`, then set __cfg.data_dir__ to your dataset path.



**Step2.** How to train?

> open `01-DL-toolbox(Pytorch)/UDL/pansharpening`

> run `python run_pansharpening.py` for training

> if you want to change the network, you could: 

1) revise arch='BDPN' in the following codes to your network name, e.g., arch='**'; 

	```python
	   import sys
           sys.path.append('../..')
           from UDL.AutoDL import TaskDispatcher
           from UDL.AutoDL.trainer import main

           if __name__ == '__main__':
           cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='BDPN')
           print(TaskDispatcher._task.keys())
           main(cfg)
	 ```
2) revise the corresponding setting in `configs/option_bdpn.py`, e.g., change 'valid_wv3.h5' to your validation data

	```python
	   cfg.eval = False, 
  
       cfg.workflow = [('train', 50), ('val', 1)], 
	
	   cfg.dataset = {'train': 'wv3', 'val': 'valid_wv3.h5'}
	```
	

**Step3.** How to test?

> open `01-DL-toolbox(Pytorch)/UDL/pansharpening`

> run `run_test_pansharpening.py` for testing

> Note you need to ensure `cfg.eval = True` or `cfg.workflow = [('val', 1)]` in the following `run_test_pansharpening.py` to run
	  

```python
	import sys
	sys.path.append('../..')
	from UDL.AutoDL import TaskDispatcher
	from UDL.AutoDL.trainer import main

	if __name__ == '__main__':
	cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='MSDCNN')
	cfg.eval = True
	cfg.workflow = [('val', 1)]
	print(TaskDispatcher._task.keys())
	main(cfg)
```

> How to get test outcome using the pretrained models?

1) find two text examples (i.e., `NY1_WV3_RR.mat`) in the path `UDL/Data/pansharpening/test_data`; 

2) load pretrained model by setting __model_path__ = "your_model_path" located in the folder of `UDL/pansharpening/models/`, or __cfg.resume_from__ = "your_model_path".

3) run `run_test_pansharpening.py`, then you may find the test results in the folder of `UDL/results`



## FAQ
**Q1.** How to customize your new network/model in this framework?

> 1) Construct your model, loss, optimizer, scheduler in `UDL/pansharpening/models/modelName_main.py` (you decide your modelName in `modelName_main.py`).

> 2) Update `UDL/pansharpening/models/__init__.py` and add `option_modelName.py`.

> 3) Config your hyperparameter in `UDL/pansharpening/configs/Option_modelName.py` (see other methods' configuration in the folder of `UDL/pansharpening/configs` for easy usage).

> 4) train your model and infer your results, see __step2__ and __step3__ for details.



**Q2.** How to customized your datasets?

You need to update:

```
01-DL-toolbox(Pytorch)/UDL/AutoDL/__init__.py.
01-DL-toolbox(Pytorch)/UDL/pansharpening/common/psdata.py.
```

**Q3.**  How to customized training settings, such as saving model, recording logs, etc.?

You need to update:

```
01-DL-toolbox(Pytorch)/UDL/mmcv/mmcv/runner/hooks
```

**Note:** Don't put model/dataset/task-related files into the folder of AutoDL.

* if you want to know more details of runner about how to train/test in `01-DL-toolbox(Pytorch)/UDL/AutoDL/trainer.py`, please see `01-DL-toolbox(Pytorch)/UDL/mmcv/mmcv/runner/epoch_based_runner.py`


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


## Acknowledgement
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- We appreciate the great contribution of [Xiao Wu](https://xiaoxiao-woo.github.io/) who is a graduate student in [UESTC](https://www.uestc.edu.cn/) to this toolbox.

## Contribution
We appreciate all contributions to improving '01-DL-toolbox(Pytorch)'. Looking forward to your contribution to DLPan-Toolbox.


## License & Copyright
This project is open sourced under GNU General Public License v3.0.

