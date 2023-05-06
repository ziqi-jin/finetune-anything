def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def load_params(model, params):
    pass


def get_opt_pamams(model, lr_list, group_keys, wd_list):
    '''

    :param model: model
    :param lr_list: list, contain the lr for each params group
    :param wd_list: list, contain the weight decay for each params group
    :param group_keys: list of list, according to the sub list to divide params to different groups
    :return: list of dict
    '''
    assert len(lr_list) == len(group_keys), "lr_list should has the same length as group_keys"
    assert len(lr_list) == len(wd_list), "lr_list should has the same length as wd_list"
    params_group = [[] for i in range(len(lr_list))]
    for name, value in model.named_parameters():
        for index, g_key in enumerate(group_keys):
            if name in g_key:
                params_group[index].append(value)
    return [{'params': params_group[i], 'lr': lr_list[i], 'weight_decay': wd_list[i]} for i in range(len(lr_list))]
