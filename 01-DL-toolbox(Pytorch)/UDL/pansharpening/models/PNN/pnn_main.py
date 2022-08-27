import torch.nn as nn
import torch.optim as optim
from .model_pnn import PNN
import numpy as np

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses

        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'Loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts


from UDL.pansharpening.models import PanSharpeningModel
class build_pnn(PanSharpeningModel, name='PNN'):
    def __call__(self, cfg):

        # important for Pansharpening models, which are from tensorflow code
        self.reg = cfg.reg


        scheduler = None

        if any(["wv" in v for v in cfg.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4
        lr = 0.0001 * 17 * 17 * spectral_num
        cfg.lr = lr
        print(f"PNN adopted another lr: {lr} in \"build_pnn in pnn_main.py\" ")


        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = PNN(spectral_num, criterion).cuda()
        target_layerParam = list(map(id, model.conv3.parameters()))
        base_layerParam = filter(lambda p: id(p) not in target_layerParam, model.parameters())

        training_parameters = [{'params': model.conv3.parameters(), 'lr': lr / 10},
                               {'params': base_layerParam}]

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  ## optimizer 2: SGD

        net_scope = 0
        for name, layer in model.named_parameters():
            if 'conv' in name and 'bias' not in name:
                net_scope += layer.shape[-1] - 1

        net_scope = np.sum(net_scope) + 1
        blk = net_scope // 2  # 8
        model.set_blk(blk)

        return model, criterion, optimizer, scheduler

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
