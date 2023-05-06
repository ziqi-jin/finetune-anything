def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
