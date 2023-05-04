from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset, VOCSemanticDataset

name_dict = {'base_det': BaseDetectionDataset, 'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset,
             'voc_sem': VOCSemanticDataset}


def get_dataset(name, path):
    if name not in name_dict:
        print('not supported dataset name, please implement it first.')
    return name_dict[name](path)
