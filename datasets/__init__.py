from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset

name_dict = {'base_det': BaseDetectionDataset, 'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
             'voc_sem': VOCSemanticDataset}


def get_dataset(name, path):
    if name not in name_dict:
        print('not supported dataset name, please implement it first.')
    return name_dict[name](path)


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
