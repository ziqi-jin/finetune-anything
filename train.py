from extend_sam import BaseExtendSam
import argparse
import OmegaConf
from torch.utils.data import DataLoader
from .datasets import BaseInstanceDataset, BaseSemanticDataset, BaseDetectionDataset

supported_tasks = {'detection': BaseDetectionDataset, 'semantic_seg': BaseSemanticDataset,
                   'instance_seg': BaseInstanceDataset}
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='instance_seg', type=str)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--cfg', default='./config/coco_instance.yaml', type=str)

if __name__ == '__main__':
    config = OmegaConf.load(parser.cfg)
    task_name = parser.task_name
    if task_name not in supported_tasks.keys():
        print("Please input the supported task name.")
    dataset = supported_tasks[task_name](parser.data_dir)
    model = BaseExtendSam()
