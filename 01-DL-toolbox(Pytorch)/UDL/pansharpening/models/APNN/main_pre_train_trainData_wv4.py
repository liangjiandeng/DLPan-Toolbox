import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_wv4 import Dataset_Pro
from model_wv4 import APNN, summaries, loss_with_l2_regularization, weights_init
from logger import create_logger, log_string
import numpy as np


import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--out_dir', metavar='DIR', default='../results',
                    help='path to save model')
parser.add_argument('--log_dir', metavar='DIR', default='logs',
                    help='path to save log')
parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                    help='useless in this script.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='APNN')

args = parser.parse_args()
args.experimental_desc = "APNN"
args.dataset = "WV4"

out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc)
print(model_save_dir)
#import shutil
#from torch.utils.tensorboard import SummaryWriter

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
""" CHANGES: 
    1. row 48:         in APNN the L1 loss is averaged only on the minibatch size,
                       for the learning rate in case the loss is averaged on minibatches, patches size,
                       and bands is lr=0.0001*17*17*nr_bands
                           a. nr_bands takes into account the number of bands
                           
    2. row 46:          <sensor> should be indicated by user here, or taken from data 
    
    3. row 46:          <nr_bands>  depends on the <sensor>
    
    4. the dataset is already normalized, so we do not need anymore <L> and <ratio> 
    
    5. row 49:          in APNN epochs=10000
    
    6. row 71:          in APNN weight_decay=0
    
    7. rows 182-195:    in pretrained_models the best PNN model is saved
"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sensor = 'WV4'
nr_bands = 4 #selected by user or taken from data?
lr = 0.0001*17*17*nr_bands
epochs = 15000
ckpt = 50
batch_size = 128
model_path = "../results/wv4/best_PNN_model_1706.pth.tar.pth.tar"
v_min = 10000
#TODO L2 norm to do where
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = APNN().cuda()
model.apply(weights_init)
if os.path.isfile(model_path):
    log_string("loading")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state"])   ## Load the pretrained Encoder
    v_min = checkpoint["loss1"]
    print("best_loss {:.7f}".format(v_min))
    log_string('APNN is Successfully Loaded from %s' % (model_path))

# summaries(model, grad=True)    ## Summary the Network
criterion = nn.L1Loss(reduction='mean').cuda()
regularization = loss_with_l2_regularization().cuda()
#用model里有的实例id去指定model中的其他参数,而不要遍历model.parameters()
target_layerParam = list(map(id, model.conv3.parameters()))
base_layerParam = filter(lambda p: id(p) not in target_layerParam, model.parameters())

training_parameters = [{'params': model.conv3.parameters(), 'lr': lr/10},
                       {'params': base_layerParam}]

optimizer = optim.SGD(training_parameters, lr=lr, momentum=0.9)

log_string("inspect optimizer setting: {}\n".format(optimizer.state_dict()))
log_string("target id: {}".format(target_layerParam))

#模型卷积层宽卷积零填充范围
# (input_size - kernel_size + 1) // 2 = 2* pad = 2 * blk = net_scope
net_scope = 0
for name, layer in model.named_parameters():
    if 'conv' in name and 'bias' not in name:
        net_scope += layer.shape[-1]-1

net_scope = np.sum(net_scope) + 1
blk = net_scope//2 #8

save_best_file = '../results/PNN/PNN_model.pth.tar'

PNN_model = {'sensor': sensor,
             'lr': lr,
             'epochs': epochs,
             'model_sampling_period': ckpt,
             'net_scope': net_scope,
             'batch_size': batch_size}


writer = SummaryWriter('../train_logs')    ## Tensorboard_show: case 2

def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader, start_epoch=0, v_min=10000):
    log_string('Start training...')

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_mae, epoch_train_mse, epoch_val_mae, epoch_val_mse = [], [], [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):

            gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            gt = gt - lms

            lms = torch.cat([lms, pan], dim=1)
            optimizer.zero_grad()  # fixed

            sr = model(lms)  # call model

            gt = gt[:, :, blk:-blk, blk:-blk]

            loss = criterion(sr, gt)  # compute loss
            new_loss = regularization(loss, model, flag=False)
            epoch_train_mae.append(loss.item())  # save all losses into a vector for one epoch

            new_loss.backward()  # fixed
            optimizer.step()  # fixed

            with torch.no_grad():
                loss = nn.MSELoss()(sr, gt)
                loss.requires_grad = False
                epoch_train_mse.append(loss.item())

        t_loss1 = np.nanmean(np.array(epoch_train_mae))  # compute the mean value of all losses, as one epoch loss
        t_loss2 = np.nanmean(np.array(epoch_train_mse))

        writer.add_scalar('mae_loss/t_mae', t_loss1, epoch)  # write to tensorboard to check
        writer.add_scalar('mae_loss/t_mse', t_loss2, epoch)
        log_string('Epoch: {}/{} training L1-loss: {:.7f}, L2-loss: {:.7f}'.format(epochs, epoch, t_loss1, t_loss2))  # print loss for each epoch
        # if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
        #     save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
                gt = gt - lms
                lms = torch.cat([lms, pan], dim=1)
                sr = model(lms)

                gt = gt[:, :, blk:-blk, blk:-blk]
                
                loss1 = criterion(sr, gt)
                loss2 = nn.MSELoss()(sr, gt)
                epoch_val_mae.append(loss1.item())
                epoch_val_mse.append(loss2.item())

            v_loss1 = np.nanmean(np.array(epoch_val_mae))
            v_loss2 = np.nanmean(np.array(epoch_val_mse))
            writer.add_scalar('val/v_mae', v_loss1, epoch)
            writer.add_scalar('val/v_mse', v_loss2, epoch)
            log_string('Epoch: {}/{} validate L1-loss: {:.7f}, L2-loss: {:7f}'.format(epochs, epoch, v_loss1, v_loss2))  # print loss for each epoch


        ### during save and simple best save ###
        # vmin = 10000
        if (epoch + 1) % ckpt == 0:
            # print("saving PNN_model_{}.pth.tar".format(epoch))
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            loss=v_loss1,
                            train_params=PNN_model),
                        '{}/PNN_model_{}.pth.tar'.format(model_save_dir, epoch + 1))

        if v_loss1 < v_min:
            if os.path.isfile(save_best_file):
                os.remove(save_best_file)
            # print("saving PNN_model.pth.tar")
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            loss1=v_loss1,
                            train_params=PNN_model),
                            '{}/best_PNN_model_{}.pth.tar'.format(model_save_dir, epoch))
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            loss1=v_loss1,
                            train_params=PNN_model),
                            '../pretrained_models/'+sensor+'_PNNplus_model.pth.tar')
            v_min = v_loss1


    writer.close()  # close tensorboard

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('../training_data/train_wv4_10000.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('../training_data/valid_wv4_10000.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader, 1707, v_min)  # call train function (call: Line 53)
