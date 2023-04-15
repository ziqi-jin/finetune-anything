from .detection import BaseDetectionDataset
from .instance_seg import BaseInstanceDataset
from .semantic_seg import BaseSemanticDataset
name_dict = {'base_det': BaseDetectionDataset, 'base_ins': BaseInstanceDataset, 'base_sem': BaseSemanticDataset}
def get_dataset(name, path):
    if name not in name_dict:
        print('not supported dataset name, please implement first.')
    return name_dict[name](path)