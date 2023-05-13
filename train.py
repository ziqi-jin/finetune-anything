'''
@copyright ziqi-jin
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from extend_sam import get_model, get_optimizer, get_scheduler, get_opt_pamams, get_runner

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='instance_seg', type=str)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--cfg', default='./config/coco_instance.yaml', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)
    train_cfg = config.train
    val_cfg = config.val_cfg
    test_cfg = config.test

    task_name = args.task_name
    if task_name not in supported_tasks:
        print("Please input the supported task name.")
    train_dataset = get_dataset(args.data_dir, 'train', train_cfg)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.bs, shuffle=True, num_workers=train_cfg.num_workers,
                              drop_last=train_cfg.drop_last)
    val_dataset = get_dataset(args.data_dir, 'val', val_cfg)
    val_loader = DataLoader(val_dataset, batch_size=val_cfg.bs, shuffle=False, num_workers=val_cfg.num_workers,
                            drop_last=val_cfg.drop_last)
    losses = get_losses(loss_names=train_cfg.loss_names)
    # according the model name to get the adapted model
    model = get_model(model_name=train_cfg.sam_name)
    opt_params = get_opt_pamams(model, lr_list=train_cfg.lr_list, group_keys=train_cfg.group_keys,
                                wd_list=train_cfg.wd_list)
    optimizer = get_optimizer(opt_name=train_cfg.opt_name)
    scheduler = get_scheduler()
    runner = get_runner(train_cfg.runner_name)(model, optimizer, losses, train_loader, val_loader, scheduler)
    # train_step
    runner.train(train_cfg)
    if test_cfg.need_test:
        runner.test(test_cfg)
