from extend_sam import BaseExtendSam
import argparse
import OmegaConf

supported_tasks = ['detection', 'semantic_seg','instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='instance_seg', type=str)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--cfg', default='./config/coco_instance.yaml', type=str)

if __name__ == '__main__':
    config = OmegaConf.load(parser.cfg)
    model = BaseExtendSam()
