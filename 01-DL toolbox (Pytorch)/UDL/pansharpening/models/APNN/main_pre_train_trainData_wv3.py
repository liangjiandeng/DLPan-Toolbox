import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_wv3 import Dataset_Pro
from model_wv3 import APNN, loss_with_l2_regularization, weights_init
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
from torchstat import stat
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


sensor = 'WV3'
nr_bands = 8  # selected by user or taken from data?
lr = 0.0001 * 17 * 17 * nr_bands
epochs = 10000
ckpt = 50
batch_size = 128
# 0.010094023841832365
model_path = "results/PNN/.pth.tar"
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = APNN().cuda()
model.apply(weights_init)

stat(model, input_size=[(9, 64, 64)])

if os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state"])  ## Load the pretrained Encoder
    print('APNN is Successfully Loaded from %s' % (model_path))
    if "loss1" in dict(checkpoint).keys():
        print("loss: {}".format(checkpoint["loss1"]))



criterion = nn.L1Loss(reduction='mean').cuda()
regularization = loss_with_l2_regularization().cuda()
target_layerParam = list(map(id, model.conv3.parameters()))
base_layerParam = filter(lambda p: id(p) not in target_layerParam, model.parameters())

training_parameters = [{'params': model.conv3.parameters(), 'lr': lr / 10},
                       {'params': base_layerParam}]

optimizer = optim.SGD(training_parameters, lr=lr, momentum=0.9, weight_decay=0)

print("inspect optimizer setting:\n", optimizer.state_dict())
print("target id:", target_layerParam)

# (input_size - kernel_size + 1) // 2 = 2* pad = 2 * blk = net_scope
net_scope = 0
for name, layer in model.named_parameters():
    if 'conv' in name and 'bias' not in name:
        net_scope += layer.shape[-1] - 1

net_scope = np.sum(net_scope) + 1
blk = net_scope // 2  # 8

save_best_file = './results/PNN/PNN_model.pth.tar'

PNN_model = {'sensor': sensor,
             'lr': lr,
             'epochs': epochs,
             'model_sampling_period': ckpt,
             'net_scope': net_scope,
             'batch_size': batch_size}

writer = SummaryWriter('./train_logs')  ## Tensorboard_show: case 2


def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/wv3/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader, start_epoch=0):
    print('Start training...')
    print(model.conv1.weight.data[0, 0, 0, 0])
    vmin = 10000
    for epoch in range(start_epoch, epochs, 1):
        flag = (epoch == (epochs - 1)) or epoch == 0
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
            new_loss = regularization(loss, model, flag=flag or (iteration == 0))
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
        print('Epoch: {}/{} training L1-loss: {:.7f}, L2-loss: {:.7f}'.format(epochs, epoch, t_loss1,
                                                                              t_loss2))  # print loss for each epoch
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
            print('Epoch: {}/{} validate L1-loss: {:.7f}, L2-loss: {:7f}'.format(epochs, epoch, v_loss1,
                                                                                 v_loss2))  # print loss for each epoch

        ### during save and simple best save ###
        # vmin = 10000
        if (epoch + 1) % ckpt == 0:
            # print("saving PNN_model_{}.pth.tar".format(epoch))
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            optim_state=optimizer.state_dict(),
                            loss=v_loss1,
                            train_params=PNN_model),
                       './results/PNN/PNN_model_{}.pth.tar'.format(epoch + 1))

        if v_loss1 < vmin:
            if os.path.isfile(save_best_file):
                os.remove(save_best_file)
            # print("saving PNN_model.pth.tar")
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            optim_state=optimizer.state_dict(),
                            loss1=v_loss1,
                            train_params=PNN_model),
                       './results/PNN/best_PNN_model_{}.pth.tar'.format(epoch))
            torch.save(dict(model=model,
                            model_state=model.state_dict(),
                            optim_state=optimizer.state_dict(),
                            loss1=v_loss1,
                            train_params=PNN_model),
                       './pretrained_models/' + sensor + '_PNNplus_model.pth.tar')
            vmin = v_loss1

    writer.close()  # close tensorboard


def fine_tune_test(file_path, training_data_loader):
    from main_test_wv3 import Tester, eval_test

    # tester = Tester(file_path, mode='ft')  # call initial model
    evaluator = Tester(file_path)
    criterion = nn.L1Loss(reduction='mean').cuda()
    " LOAD PRETRAINED MODEL"
    model_path = "./results/PNN/.pth.tar"
    if os.path.isfile(model_path):
        print("loading model")
        checkpoint = torch.load(model_path)
        # checkpoint = torch.load('./pretrained_models/' + sensor_model)
        print(checkpoint.keys())
        net = checkpoint['model']
        print(net.conv1.weight.data[0, 0, 0, 0])
        net.load_state_dict(checkpoint['model_state'])
        train_params = checkpoint['train_params']
        lr = train_params['lr']  # learning rate
        print("lr", lr)
        FT_epochs = 1000  # number of fine tuning epochs

    else:
        lr = 0.0001 * 17 * 17 * nr_bands
        FT_epochs = epochs
        net = APNN().cuda()
        print(net.conv1.weight.data[0, 0, 0, 0])
    '''
    tensor(-0.0003, device='cuda:0')
    tensor(-0.0204, device='cuda:0')
    '''
    print(net.conv1.weight.data[0, 0, 0, 0])
    print(net)
    # print("pretrain loss: ", checkpoint["loss1"])




    print(dict(net.named_parameters()).keys())
    target_layerParam = list(map(id, net.conv3.parameters()))
    base_layerParam = filter(lambda p: id(p) not in target_layerParam, net.parameters())
    training_parameters = [{'params': net.conv3.parameters(), 'lr': lr/10},
                           {'params': base_layerParam}]

    optimizer = optim.SGD(training_parameters, lr=lr, momentum=0.9, weight_decay=0)

    try:
        optimizer.load_state_dict(checkpoint["optim_state"])
    except:
        print("default optim_state")

    v_min = 10000
    ft_loss = np.zeros(FT_epochs)

    eval_test(net, evaluator, mode="eval", mode2="pre")#0.0114576
    for epoch in range(FT_epochs):
        net.train()
        epoch_train_mae = []
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, lms, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            gt = gt - lms

            lms = torch.cat([lms, pan], dim=1)
            optimizer.zero_grad()  # fixed

            sr = net(lms)  # call model

            gt = gt[:, :, blk:-blk, blk:-blk]

            loss = criterion(sr, gt)  # compute loss
            new_loss = regularization(loss, net, flag=False)

            epoch_train_mae.append(loss.item())  # save all losses into a vector for one epoch

            new_loss.backward()  # fixed
            optimizer.step()  # fixed

        running_loss = np.nanmean(epoch_train_mae)
        ft_loss[epoch] = running_loss

        if running_loss < v_min:
            v_min = running_loss
            PATH = './ft_network/'
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            torch.save(dict(model=net,
                            model_state=net.state_dict(),
                            loss=ft_loss),
                       PATH + '/net.pth.tar')
            net.eval()
            eval_test(net, evaluator, mode="eval", mode2="ft")
        print('[%d] loss: %.20f' % (epoch + 1, running_loss))


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
    # file_path = "./test_data/TestData_wv3.h5"
    # fine_tune_test(file_path, training_data_loader)

'''

'''
