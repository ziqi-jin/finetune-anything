'''
@copyright ziqi-jin
'''
from extend_sam import BaseExtendSam
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from .datasets import get_dataset
from .utils import get_losses
from .extend_sam import get_model, get_optimizer, get_scheduler, BaseRunner, get_opt_pamams

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='instance_seg', type=str)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--cfg', default='./config/coco_instance.yaml', type=str)

if __name__ == '__main__':
    config = OmegaConf.load(parser.cfg)
    train_cfg = config.train
    task_name = parser.task_name
    if task_name not in supported_tasks:
        print("Please input the supported task name.")
    dataset = get_dataset(train_cfg.dataset_name, parser.data_dir)
    data_loader = DataLoader(dataset, batch_size=train_cfg.bs, shuffle=True, num_workers=train_cfg.num_workers,
                             drop_last=train_cfg.drop_last)
    losses = get_losses(loss_names=train_cfg.loss_names)
    # according the model name to get the adapted model
    model = get_model(model_name=train_cfg.sam_name)
    opt_params = get_opt_pamams(model, lr_list=train_cfg.lr_list, group_keys=train_cfg.group_keys, wd_list=train_cfg.wd_list)
    optimizer = get_optimizer(opt_name=train_cfg.opt_name)
    scheduler = get_scheduler()
    # train_step
    runner = BaseRunner()
