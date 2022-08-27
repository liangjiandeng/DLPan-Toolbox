import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_wv2 import Dataset_Pro
import h5py
from data_single_read import load_set
from evaluate import compute_index
from model_wv2 import APNN, summaries, weights_init, loss_with_l2_regularization
import numpy as np
import scipy.io as sio
from time import time
from evaluate import analysis_accu


###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################

def load_set(file_path, blk):

    suffix = file_path.split('.')

    if suffix[-1] == 'h5':
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
        shape_size = len(ms1.shape)
    elif suffix[-1] == 'mat':
        # ===== case2: HxWxC
        data = sio.loadmat(file_path)  #
        print(data.keys())
        ms1 = data["I_MS_LR"][...]  # NxCxHxW=0,1,2,3
        shape_size = len(ms1.shape)
    else:
        print("file format is not supporetd")
        raise NotImplemented

    if suffix[-2][-2:] == 'FR':
        data['I_GT'] = data['I_MS_LR'] #exception

    if shape_size == 4:  # NxCxHxW
        # tensor type:
        lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0
        lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # NxCxHxW = 4x8x512x512
        pan1 = np.array(pan1, dtype=np.float32) / 2047.0
        pan = torch.from_numpy(pan1)

        test_I_in1 = np.concatenate([lms1, pan1], axis=1)  # NxCxHxW = Nx9xHxW
        test_I_in1 = np.pad(test_I_in1, ((0, 0), (0, 0), (blk, blk), (blk, blk)), mode='edge')  # pading
        test_I_in = torch.from_numpy(test_I_in1)  # NxCxHxW = Nx9xHxW

        ms1 = data['ms'][...]  # NxCxHxW = 4x8x512x512
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0
        ms = torch.from_numpy(ms1)

        gt1 = data['gt'][...]  # NxCxHxW = 4x8x512x512
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0
        gt = torch.from_numpy(gt1)

        return test_I_in, ms, pan, gt

    if shape_size == 3:  # HxWxC

        # tensor type:
        lms1 = data['I_MS'][...]  # HxWxC=0,1,2
        lms1 = np.expand_dims(lms1, axis=0)  # 1xHxWxC
        lms1 = np.array(lms1, dtype=np.float32) / 2047.0  # 1xHxWxC
        lms = torch.from_numpy(lms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        pan1 = data['I_PAN'][...]  # HxW
        pan1 = np.expand_dims(pan1, axis=0)  # 1xHxW
        pan1 = np.expand_dims(pan1, axis=3)  # 1xHxWx1
        pan1 = np.array(pan1, dtype=np.float32) / 2047.  # 1xHxWx1
        pan = torch.from_numpy(pan1).permute(0, 3, 1, 2)  # Nx1xHxW:

        test_I_in1 = np.concatenate([lms1, pan1], axis=3)  # 1xHxWx(C+1) = Nx9xHxW
        test_I_in1 = np.transpose(test_I_in1, (0, 3, 1, 2))  # 1x(C+1)xHxW
        test_I_in1 = np.pad(test_I_in1, ((0, 0), (0, 0), (blk, blk), (blk, blk)), mode='edge')  # NCHW
        test_I_in = torch.from_numpy(test_I_in1)  # NxCxHxW = Nx9xHxW

        ms1 = data['I_MS_LR'][...]  # HxWxC=0,1,2
        ms1 = np.expand_dims(ms1, axis=0)  # 1xHxWxC
        ms1 = np.array(ms1, dtype=np.float32) / 2047.0  # 1xHxWxC
        ms = torch.from_numpy(ms1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        gt1 = data['I_GT'][...]  # HxWxC=0,1,2
        gt1 = np.expand_dims(gt1, axis=0)  # 1xHxWxC
        gt1 = np.array(gt1, dtype=np.float32) / 2047.0  # 1xHxWxC
        gt = torch.from_numpy(gt1).permute(0, 3, 1, 2)  # NxCxHxW  or HxWxC

        return test_I_in, ms, pan, gt


###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################

## 1) initial test by model ##
blk = 8


class Tester():
    def __init__(self, file_path, mode):

        test_I_in, test_ms, test_pan, test_gt = load_set(file_path, blk)
        self.test_I_in = test_I_in
        self.test_ms = test_ms
        self.test_pan = test_pan
        self.test_gt = test_gt
        self.mode = mode
        self.file_path = file_path
        "the fine tuning phase requires downgraded input resolution"
        if mode == 'ft':
            from wald_utilities import wald_protocol
            ms_lr, pan_lr = wald_protocol(test_ms, test_pan, 4., 'WV2', channels=8)
            self.test_I_in = torch.cat((ms_lr, pan_lr), dim=1)
            self.test_I_in = torch.nn.functional.pad(self.test_I_in, (8, 8, 8, 8), mode='reflect')  # NCHW
            self.test_gt = self.test_ms

    def __call__(self, model):
        x = self.test_I_in  # send to cuda, important!
        x = x.cuda().float()  # convert to tensor type:
        out2 = model(x)

        if self.mode == 'test' or self.mode == 'RR':
            sr = out2 + self.test_I_in[:, :-1, blk:-blk, blk:-blk].cuda()  # NxCxHxW
            sr = sr.permute(0, 2, 3, 1)
            gt = self.test_gt.permute(0, 2, 3, 1).cuda()
            our_CC, our_PSNR, our_SSIM, our_SAM, our_ERGAS = analysis_accu(gt[0, ...], sr[0, ...], 4)
            print(f'[{self.file_path}]: our_CC: {our_CC}, our_PSNR: {our_PSNR}, '
                  f'our_SSIM: {our_SSIM},\n'
                  f'our_SAM: {our_SAM} our_ERGAS: {our_ERGAS}')

        return out2


## 2) target-adative's fine_tune_training, i.e., PNNplus##

def test(file_path, sensor_model):

    suffix = file_path.split('.')[-2][-2:]
    if suffix== 'RR' or suffix == 'FR':
        simulated = suffix
    else:
        simulated = 'test'

    tester = Tester(file_path, mode='ft')  # call initial model
    evaluator = Tester(file_path, mode=simulated)
    criterion = nn.L1Loss(reduction='mean').cuda()
    regularization = loss_with_l2_regularization().cuda()
    " LOAD PRETRAINED MODEL"
    init_loss = 0
    model_path = "../pretrained_models/1WV2_PNNplus_model.pth.tar"
    if os.path.isfile(model_path):
        print("loading model")
        checkpoint = torch.load(model_path)
        # checkpoint = torch.load('./pretrained_models/' + sensor_model)
        print(checkpoint.keys())
        net = checkpoint['model']
        print(net.conv1.weight.data[0, 0, 0, 0])
        net.load_state_dict(checkpoint['model_state'])
        lr_ = 1e-6
        FT_epochs = 200  # number of fine tuning epochs
        # init_loss = checkpoint["loss1"]
    else:
        nr_bands = 8  # selected by user or taken from data?
        lr_ = 0.0001 * 17 * 17 * nr_bands
        FT_epochs = 5000
        net = APNN().cuda()
        net.apply(weights_init)
        print(net.conv1.weight.data[0, 0, 0, 0])

    print(net.conv1.weight.data[0, 0, 0, 0])
    print(net)

    test_gt = tester.test_gt - tester.test_I_in[:, :-1, blk:-blk, blk:-blk]
    pretrain_inIt_loss = criterion(net(tester.test_I_in.cuda()), test_gt.cuda()).item()
    print("init loss: {:.20f} pretrain_inIt loss: {:.20f}".format(init_loss, pretrain_inIt_loss))
    eval_test(net, evaluator, mode="eval", mode2="pre")
    print('-' * 100)
    "scaling learning rate on last layer"
    target_layerParam = list(map(id, net.conv3.parameters()))
    base_layerParam = filter(lambda p: id(p) not in target_layerParam, net.parameters())

    training_parameters = [{'params': net.conv3.parameters(), 'lr': lr_ / 10},
                           {'params': base_layerParam}]

    optimizer = optim.SGD(training_parameters, lr=lr_, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_min = 10000
    ft_loss = np.zeros(FT_epochs)
    Train_time = time()
    print(tester.test_gt.shape)
    ## 2.1) "FINE TUNING"--training
    for epoch in range(FT_epochs):  # loop over the testing image multiple times
        net.train()
        # running_loss = 0.0
        # loading testing image
        test_I_in = tester.test_I_in
        test_gt = tester.test_gt

        # residual
        test_gt = test_gt - test_I_in[:, :-1, blk:-blk, blk:-blk]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x1 = test_I_in  # send to cuda, important!
        x2 = test_gt
        x1 = x1.cuda().float()  # convert to tensor type:
        x2 = x2.cuda().float()  # convert to tensor type:

        outputs = net(x1)

        loss = criterion(outputs, x2)  # compute loss
        new_loss = regularization(loss, net, flag=False)

        new_loss.backward()
        optimizer.step()

        running_loss = loss.item()
        ft_loss[epoch] = running_loss

        if running_loss < v_min:

            PATH = '../ft_network/WV2'
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            torch.save(dict(model=net,
                            model_state=net.state_dict(),
                            loss=ft_loss),
                       PATH + '/net.pth.tar')
            if np.abs(running_loss - v_min) > 1e-3:
                net.eval()
                eval_test(net, evaluator, mode="eval", mode2="ft")
            v_min = running_loss

        print('[%d] loss: %.20f' % (epoch + 1, running_loss))

    Train_time = time() - Train_time

    ## 2.2) "FINE TUNING"--testing
    " LOAD BEST MODEL"
    checkpoint = torch.load('../ft_network/WV2/net.pth.tar')
    net = checkpoint['model']
    net.load_state_dict(checkpoint['model_state'])

    " PANSHARPENING "
    net.to(device)
    net.eval()
    print(net.conv3.weight.data[0, 0, 0, 0], net.conv2.weight.data[0, 0, 0, 0], net.conv1.weight.data[0, 0, 0, 0])
    eval_test(net, evaluator, mode="eval", mode2="ft")


def eval_test(net, evaluator, mode="pre", mode2="pre"):
    with torch.no_grad():
        Test_time = time()
        sr = evaluator(net)  # NxCxHxW
        Test_time = time() - Test_time

        # skip connection to add low resolution ms and residual(np version)
        sr = sr.cpu().detach().numpy() + evaluator.test_I_in[:, :-1, blk:-blk,
                                         blk:-blk].cpu().detach().numpy()  # NxCxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.from_numpy(sr)  # convert to tensor version
        sr = sr.permute(0, 2, 3, 1).cpu().detach().numpy()  # to: NxHxWxC
        sr = np.clip(sr, 0, 1)

        # print('------>  [PNN+]: Fine-tuning (%d it) time = %0.4f  //  Prediction time = %0.4f' % (
        # FT_epochs, Train_time, Test_time))

        num_exm = sr.shape[0]
        if num_exm == 1:
            if evaluator.mode == "RR":
                file_name = "apnn_wv2_rs" + '_rio_' + mode2 + ".mat"
            if evaluator.mode == "FR":
                file_name = "apnn_wv2_os" + '_rio_' + mode2 + ".mat"

            # file_name2 = "E:/01-DL-Pansharpening-Toolbox/03-Comparisons(Matlab)/2_DL_Result/WV4/APNN"
            file_name2 = "../results"
            save_name = os.path.join(file_name2, file_name)
            sio.savemat(save_name, {'apnn_wv2': sr[0, :, :, :]})
        else:
            for index in range(num_exm):  # save the DL results to the 03-Comparisons(Matlab)
                file_name = "apnn_wv2_rs" + str(index) + mode2 + ".mat"
                # file_name2 = "E:/01-DL-Pansharpening-Toolbox/03-Comparisons(Matlab)/2_DL_Result/WV4/APNN"
                file_name2 = "../results"
                save_name = os.path.join(file_name2, file_name)
                sio.savemat(save_name, {'apnn_wv2_rs': sr[index, :, :, :]})


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "../test_data/imgs/Rio_WV2_FR.mat"
    # file_path = "../test_data/TestData_wv2.h5"
    "SELECT SENSOR AND TESTING IMAGE"
    sensor_model = 'WV2'
    available_models = ['IKONOS', 'GeoEye1', 'WV2', 'WV3', 'WV4', 'QB']
    if sensor_model in available_models:
        sensor_model = sensor_model + '_PNNplus_model.pth.tar'

    test(file_path, sensor_model)
