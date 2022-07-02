import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt


class lr_scheduler(object):

    def __init__(self, lr, epochs):
        self.epochs = epochs
        self.lr = lr
        self.lr_scheduler = None

    # 六大学习率调整策略，lr = lr * gamma
    '''
    ReduceLROnPlateau:
        mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
        factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
        patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
        verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
        threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
        当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
        当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
        当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
        当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best - threshold；
        threshold(float)- 配合 threshold_mode 使用。
        cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
        min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置
    '''

    def set_optimizer(self, optimizer, lr_scheduler):
        self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        # self.scheduler = []
        if lr_scheduler == torch.optim.lr_scheduler.StepLR:
            # 等间距阶段式衰减
            self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        elif lr_scheduler == optim.lr_scheduler.ReduceLROnPlateau:
            # Reduce learning rate when validation accuarcy plateau.
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
        elif lr_scheduler == optim.lr_scheduler.MultiStepLR:
            # milestones=[epoch1,epoch2,...] 阶段式衰减
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300],
                                                               gamma=0.5)  # [50, 100, 150, 200, 250, 300, 350, 400], gamma=0.5)
        elif lr_scheduler == optim.lr_scheduler.ExponentialLR:
            # 指数衰减x, 0.1,0.01,0.001,...
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
        elif lr_scheduler == optim.lr_scheduler.CosineAnnealingLR:
            # Cosine annealing learning rate.
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)
        elif lr_scheduler == optim.lr_scheduler.CyclicLR:
            self.lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-4, step_size_down=30,
                                                            step_size_up=150, cycle_momentum=False)
        elif lr_scheduler == optim.lr_scheduler.LambdaLR:
            # 学习率 = 初始学习率 * lr_lambda(last_epoch）
            curves = lambda epoch: epoch // 30
            # lambda2 = lambda epoch: 0.95 ** epoch
            # lr_lambda对应optimizer中的keys，model.parameters()就只有一个lambda函数
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[curves])
        elif lr_scheduler == optim.lr_scheduler.CosineAnnealingWarmRestarts:
            # To 初始周期
            # T_mult 每次循环 周期改变倍数  T_0 = T_0*T_mult
            # Learning rate warmup by 10 epochs.
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        else:
            print("self.lr_scheduler not in pytorch")

    def adjust_2_learning_rate(self, epoch):
        """编写2种形式的学习率衰减策略的组合"""
        param_groups = self.optimizer.param_groups
        if epoch <= 5:
            lr = [param_groups[0]['lr'] * 0.9]
            for param_group, val in zip(param_groups, lr):
                param_group['lr'] = val
        else:
            for param_group in param_groups:
                if epoch % 5 == 0:
                    # 0.09 0.009 0.0009
                    param_group['lr'] *= 0.9
        # print(param_group['lr'])

    def adjust_1_learning_rate(self, epoch, mini_lr=1e-6):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if self.optimizer.param_groups[0]["lr"] < mini_lr:
            lr = 1e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        if epoch <= 40:  # 40 20 80
            # lr = self.lr
            lr = self.lr * (0.1 ** (epoch // 20))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        elif epoch == 81:  # 41
            self.lr = self.optimizer.param_groups[0]["lr"]
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = 1e-4
        # if epoch >= 42 and epoch % 5 ==0:
        if epoch >= 81:
            lr = self.lr * (0.9 ** (epoch // 20))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        elif epoch == 81:
            lr = 1e-5
        else:
            lr = 1e-5
            # self.lr = self.lr * (0.9 ** (epoch // 50))
        # #if epoch <= 120:
        #     lr = self.lr * (0.9 ** (epoch // 50))
        # elif epoch == 121:
        #    self.lr = self.optimizer.param_groups[0]["lr"]
        #    lr = self.lr * (0.9 ** (epoch // 50))
        # else:
        #    lr = 0.01
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self, epoch):
        # if not end:
        #     self.optimizer.step()
        # else:
        if self.lr_scheduler is None:
            # self.optimizer.step()
            self.adjust_1_learning_rate(epoch)
        else:
            # self.optimizer.step()
            self.lr_scheduler.step(epoch)

    # preprint lr_map
    def get_lr_map(self, title, out_file=None, viz=False):
        plt.figure()
        lr = []
        print("preprint lr_scheduler")
        tmp = self.optimizer.param_groups[0]['lr']
        if self.lr_scheduler is None:
            for epoch in range(self.epochs):
                self.step(epoch)
                # TODO:按层绘制
                # print(self.optimizer.param_groups[0]['lr'])
                lr.append(self.optimizer.param_groups[0]['lr'])
        else:
            for epoch in range(self.epochs):
                self.step(epoch)
                try:
                    lr.append(self.lr_scheduler.get_last_lr())
                    # lr.append(self.lr_scheduler.get_lr())
                except:
                    # ReduceLROnPlateau没有get_lr方法
                    lr.append(self.optimizer.param_groups[0]['lr'])
        plt.plot(list(range(self.epochs)), lr)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title(title)
        if out_file is not None:
            plt.savefig(out_file)
        if viz:
            plt.show()
        self.optimizer.param_groups[0]['lr'] = tmp
        self.lr = tmp


def tune_param():
    ...


def partial_train(model, layers: list):
    # forzen layers
    for param in model.parameters():
        if layers is not None and layers in param:
            continue
        param.requires_grad = False

    # Replace the last fc layer
    model.fc = nn.Linear(512, 100)
    return model


if __name__ == "__main__":
    from torchvision.models import AlexNet
    import matplotlib.pyplot as plt

    model = AlexNet(num_classes=2)


    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear1 = nn.Linear(1, 5)  # input and output is 1 dimension
            self.linear2 = nn.Linear(5, 1)

        def forward(self, x):
            out = self.linear1(x)
            out = self.linear2(out)
            return out


    glm = LinearRegression()

    optimizer = optim.SGD(params=glm.parameters(), lr=0.1)

    epochs = 450
    # 构造一个带warmup小学习率的optimizer，再上升到标准值，再正常周期下降
    lrs = lr_scheduler(0.1, epochs)
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.MultiStepLR)
    # lrs.get_lr_map("MultiStepLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.ExponentialLR)
    # lrs.get_lr_map("ExponentialLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.StepLR)
    # lrs.get_lr_map("StepLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CyclicLR)
    # lrs.get_lr_map("CyclicLR")
    # # lrs.set_optimizer(optimizer, optim.lr_scheduler.ReduceLROnPlateau)
    # # lrs.get_lr_map("ReduceLROnPlateau")
    lrs.set_optimizer(optimizer, None)
    lrs.get_lr_map("LambdaLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CosineAnnealingLR)
    # lrs.get_lr_map("CosineAnnealingLR")
    # lrs.set_optimizer(optimizer, optim.lr_scheduler.CosineAnnealingWarmRestarts)
    # lrs.get_lr_map("CosineAnnealingWarmRestarts")
