# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import glob
import torch
from torch.utils.data import DataLoader


class PansharpeningSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        # self.patch_size = args.patch_size
        self.writers = {}
        self.args = args


    def get_dataloader(self, dataset_name, distributed):

        if any(list(map(lambda x: x in dataset_name, ['wv2', 'wv3', 'wv4', 'qb', 'gf']))):
            if "hp" in dataset_name:
                # high-pass filter
                from UDL.pansharpening.common.dataset_hp import Dataset_Pro
                dataset_name = dataset_name.split('_')[0] #'wv2_hp'
                dataset = Dataset_Pro('/'.join([self.args.data_dir, 'training_data', f'train_{dataset_name}.h5']), img_scale=self.args.img_range)
            else:

                from UDL.pansharpening.common.dataset import Dataset_Pro
                dataset = Dataset_Pro('/'.join([self.args.data_dir, 'training_data', f'train_{dataset_name}.h5']), img_scale=self.args.img_range)

        else:
            print(f"train_{dataset_name} is not supported.")
            raise NotImplementedError


        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = \
            DataLoader(dataset, batch_size=self.samples_per_gpu,
                       persistent_workers=(True if self.workers_per_gpu > 0 else False), pin_memory=True,
                       shuffle=(sampler is None), num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return dataloaders, sampler

    def get_test_dataloader(self, dataset_name, distributed):
        # creat data for validation
        if dataset_name in ['wv3', 'wv2', 'qb', 'gf2']:
            from UDL.pansharpening.common.dataset_hp import Dataset_Pro
            dataset = Dataset_Pro(
                '/'.join([self.args.data_dir, 'validation_data', f'valid_{dataset_name}.h5']), img_scale=self.args.img_range)
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders = \
                DataLoader(dataset, batch_size=self.samples_per_gpu, pin_memory=True,
                           shuffle=False, num_workers=self.workers_per_gpu, drop_last=True, sampler=sampler)

        return self.dataloaders, sampler

    def get_eval_dataloader(self, dataset_name, distributed):
        if 'new_data' in dataset_name:
            from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
            dataset = SingleDataset(['/'.join([self.args.data_dir, "test_data", f"{dataset_name}.mat"])], dataset_name, img_scale=self.args.img_range)

        elif 'multi_exm1258' in dataset_name:
            from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
            dataset = MultiExmTest_h5('/'.join([self.args.data_dir, "test_data/WV3_Simu_mulExm/test1_mulExm1258.mat"]), dataset_name, suffix='.mat', img_scale=self.args.img_range)

        elif 'singleMat' in dataset_name:
            from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
            dataset = SingleDataset(glob.glob('/'.join([self.args.data_dir, "test_data", "*.mat"])), dataset_name)

        elif 'Test(HxWxC)' in dataset_name:
            # Test(HxWxC)_gf2_data_fr/rr...
            from UDL.pansharpening.evaluation.ps_evaluate import SingleDatasetV2
            satellite = dataset_name.split('_')[1]
            type = 'FR-Data' if 'fr' in dataset_name else 'RR-Data'
            dataset = SingleDatasetV2(glob.glob('/'.join([self.args.data_dir, f"/test_data/{satellite.upper()}/{type}/*.mat"])), dataset_name, img_scale=self.args.img_range)

        elif 'multiExm' in dataset_name:
            satellite = dataset_name.split('_')[0]
            suffix = dataset_name.split('.')[-1]
            from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
            dataset = MultiExmTest_h5('/'.join([self.args.data_dir, f"test_data/{satellite.upper()}/test_{dataset_name}"]),
                                      dataset_name, suffix=f'.{suffix}', img_scale=self.args.img_range)
        # elif 'multiExm' in dataset_name or mode == 'val':
        #
        #     if mode == "val" and dataset_name in self.mapping:
        #         dataset_name = self.mapping[dataset_name]

            # satellite = dataset_name.split('_')[0]
            # suffix = dataset_name.split('.')[-1]
            # from UDL.pansharpening.evaluation.ps_evaluate import MultiExmTest_h5
            # dataset = MultiExmTest_h5('/'.join([self.args.data_dir, f"test_data/{satellite.upper()}/test_{dataset_name}"]),
            #                           dataset_name, suffix=f'.{suffix}', img_scale=self.args.img_range)
        elif 'Test' in dataset_name:
            from UDL.pansharpening.evaluation.ps_evaluate import SingleDataset
            satellite = dataset_name.split('_')[0]
            dataset = SingleDataset(glob.glob('/'.join([self.args.data_dir, "test_data", satellite, "01-Test_Single", "RR-Data", "/*.mat"])), dataset_name, img_scale=self.args.img_range)




        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # if not dataset_name in self.dataloaders:
        dataloaders = \
            DataLoader(dataset, batch_size=1,
                       shuffle=False, num_workers=1, drop_last=False, sampler=sampler)
        return dataloaders, sampler



if __name__ == '__main__':
    # from option import args
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.samples_per_gpu = 8
    args.workers_per_gpu = 0
    args.data_dir = "C:/Datasets/pansharpening_2"
    args.dataset = 'gf2'

    # survey
    # wv3 9714 16-64
    # wv2 15084 16-64
    # gf2 19809 16-64
    # qb  17139 16-64
    sess = PansharpeningSession(args)
    train_loader, _ = sess.get_test_dataloader(args.dataset, False)
    print(len(train_loader))

    # import scipy.io as sio
    #
    # x = sio.loadmat("D:/Datasets/pansharpening/training_data/train1.mat")
    # print(x.keys())


