from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset
from .transforms import get_transforms
name_dict = {'base_det': BaseDetectionDataset, 'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
             'voc_sem': VOCSemanticDataset}


def get_dataset(path, mode, cfg):

    name = cfg.dataset_name
    if name not in name_dict:
        print('not supported dataset name, please implement it first.')
    # TODO customized dataset params:
    # customized dataset params example:
    # if xxx:
    #   param1 = cfg.xxx
    #   param2 = cfg.xxx
    # return name_dict[name](path, model, param1, param2, ...)
    transform = get_transforms(cfg.transfomrs)
    return name_dict[name](path, mode, transform)


class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)

    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)

        return data

