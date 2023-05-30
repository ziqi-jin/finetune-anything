import torchvision.transforms as T

AVIAL_TRANSFORM = {'resize': T.Resize, 'to_tensor': T.ToTensor}


def get_transforms(T_names):
    T_list = []
    for T_name in T_names:
        assert T_name in AVIAL_TRANSFORM, "{T_name} is not supported transform, please implement it and add it to " \
                                          "AVIAL_TRANSFORM first.".format(T_name=T_name)
        T_list.append(AVIAL_TRANSFORM[T_name])
    return T.Compose(T_list)
