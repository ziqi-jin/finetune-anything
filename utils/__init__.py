import torch.nn as nn

name_dict = {'ce_loss': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss}


def get_losses(loss_names):
    loss_dict = {}
    for name in loss_names:
        if name not in name_dict:
            print('not supported loss name, please implement it first.')
            break
        loss_dict[name] = name_dict[name]
    if len(loss_dict) != len(loss_names):
        return None
    return loss_dict
