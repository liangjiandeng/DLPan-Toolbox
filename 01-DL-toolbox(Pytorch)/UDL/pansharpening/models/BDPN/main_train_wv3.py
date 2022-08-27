# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, Ran Ran, LiangJian Deng
# @reference:
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_wv3 import Dataset_Pro
from model_wv3 import BDPN
from torchstat import stat
import numpy as np
from tensorboardX import SummaryWriter
import shutil
from loss_utils import compute_charbonnier_loss, compute_ergas_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0001
epochs = 1000
ckpt = 50
batch_size = 8
lambda_v = 1.0
lambda_init = 0.05
lambda_declay = 5
model_path = "Weights/wv3/.pth"

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = BDPN().cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('PANnet is Successfully Loaded from %s' % (model_path))

stat(model, input_size=[(8, 16, 16), (1, 64, 64)])
#criterion = nn.MSELoss(size_average=True).cuda()

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  # optimizer 2
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)   # optimizer 1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100,
                                               gamma=0.8)  # lr = lr* gamma for every step_size(epochs) = 180

# ============= 4) Tensorboard_show + Save_model ==========#
#if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
#   shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs

writer = SummaryWriter('./train_logs')    ## Tensorboard_show: case 2

def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    global lambda_v
    print('Start training...')

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        if epoch <= 100:
            lambda_v = 1.0 - lambda_init*(epoch//lambda_declay)  # decrease lambda_v for every lambda_declay epochs

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

            optimizer.zero_grad()  # fixed

            sr, sr_down = model(ms, pan)  # call model: sr=4x64x64; sr_down=4x32x32

            gt_down = F.interpolate(gt, scale_factor=0.5, mode='nearest')   # nearest down 2
            loss1 = compute_charbonnier_loss(sr_down, gt_down)  # compute loss1; orig: loss = criterion(sr, gt)
            loss2 = compute_charbonnier_loss(sr, gt)  # compute loss2

            loss = lambda_v*loss1 + (1.0 - lambda_v)*loss2   # total loss:
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed

            # for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                # writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

        lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss (lr={}, lam_v={}): {:.7f}'.format(epochs, epoch, lr_scheduler.get_last_lr(), lambda_v, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

                sr, sr_down = model(ms, pan)  # call model

                gt_down = F.interpolate(gt, scale_factor=0.5, mode='nearest')  # nearest down 2
                loss1 = compute_charbonnier_loss(sr_down, gt_down)  # compute loss1; orig: loss = criterion(sr, gt)
                loss2 = compute_charbonnier_loss(sr, gt)  # compute loss2

                loss = lambda_v * loss1 + (1.0 - lambda_v) * loss2

                epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('./training_data/train_wv3_10000.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('./training_data/valid_wv3_10000.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 53)
