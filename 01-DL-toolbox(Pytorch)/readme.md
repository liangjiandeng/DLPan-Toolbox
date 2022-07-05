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
│   │   │   ├── test_wv3_multiExm.h5
│   │   │   ├── test_wv3_multiExm.h5
│   │   │   ├── ...
```

* Check and revise your dataset path in `01-DL-toolbox(Pytorch)/UDL/Basis/option.py` (may not need to revise), or you can print the output of `run_pansharpening.py`, then set __cfg.data_dir__ to your dataset path.



**Step2.** How to train?

> Open `01-DL-toolbox(Pytorch)/UDL/pansharpening`,  run the following code for training:

> `python run_pansharpening.py`

> If you want to change the network, you could: 

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
	 ````
2) revise the corresponding setting in `configs/option_bdpn.py`, e.g., change 'valid_wv3.h5' to your validation data

	```
	   cfg.eval = False, 
  
       cfg.workflow = [('train', 50), ('val', 1)], 
	
	   cfg.dataset = {'train': 'wv3', 'val': 'valid_wv3.h5'}
	```
	

**Step3.** How to test?

> Open `01-DL-toolbox(Pytorch)/UDL/pansharpening`,  run the following code for testing:

> `run_test_pansharpening.py`

> Note you need to ensure `cfg.eval = True` or `cfg.workflow = [('val', 1)]` in `run_test_pansharpening.py` to run
	  
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



**Step4**. How to customize the code.

One model is divided into three parts:

1. Record hyperparameter configurations in folder of `01-DL-toolbox(Pytorch)/UDL/pansharpening/configs/Option_modelName.py`. For example, you can load pretrained model by setting __model_path__ = "your_model_path" or __cfg.resume_from__ = "your_model_path".

2. Set model, loss, optimizer, scheduler in folder of `01-DL-toolbox(Pytorch)/UDL/pansharpening/models/modelName_main.py`.

3. Write a new model in folder of `01-DL-toolbox(Pytorch)/UDL/pansharpening/models/*modelName*/model_modelName.py`.

Note that when you add a new model into `01-DL-toolbox(Pytorch)`, you need to update `01-DL-toolbox(Pytorch)/UDL/pansharpening/models/__init__.py` and add option_modelName.py.

**Others**
* if you want to add customized datasets, you need to update:

```
01-DL-toolbox(Pytorch)/UDL/AutoDL/__init__.py.
01-DL-toolbox(Pytorch)/UDL/pansharpening/common/psdata.py.
```

* if you want to add customized tasks, you need to update:

```
1.Put model_newModelName and newModelName_main in 01-DL-toolbox(Pytorch)/UDL/taskName/models.
2.Create a new folder of 01-DL-toolbox(Pytorch)/UDL/taskName/configs to put option_newModelName.
3.Update 01-DL-toolbox(Pytorch)/UDL/AutoDL/__init__.p.
4.Add a class in 01-DL-toolbox(Pytorch)/UDL/Basis/python_sub_class.py, like this:
class PanSharpeningModel(ModelDispatcher, name='pansharpening'):
```

* if you want to add customized training settings, such as saving model, recording logs, and so on. you need to update:

```
01-DL-toolbox(Pytorch)/UDL/mmcv/mmcv/runner/hooks
```

Note that: Don't put model/dataset/task-related files into the folder of AutoDL.

* if you want to know more details of runner about how to train/test in `01-DL-toolbox(Pytorch)/UDL/AutoDL/trainer.py`, please see `01-DL-toolbox(Pytorch)/UDL/mmcv/mmcv/runner/epoch_based_runner.py`

## Contribution
We appreciate all contributions to improving '01-DL-toolbox(Pytorch)'. Looking forward to your contribution to DLPan-Toolbox.


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


## License & Copyright
This project is open sourced under GNU General Public License v3.0.

