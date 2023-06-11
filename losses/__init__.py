import torch.nn as nn
from .losses import CustormLoss

name_dict = {'ce': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
             'test_custom': CustormLoss, 'mse': nn.MSELoss}


def get_losses(losses):
    loss_dict = {}
    for name in losses:
        assert name in name_dict, print('{name} is not supported, please implement it first.'.format(name=name))
        if losses[name].params is not None:
            loss_dict[name] = name_dict[name](**losses[name].params)
        else:
            loss_dict[name] = name_dict[name]()
    return loss_dict
