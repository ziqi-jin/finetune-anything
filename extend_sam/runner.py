from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log
import torch
import cv2


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()


class SemRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']

    def train(self, cfg):
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_mIoU = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        writer = None
        if cfg.use_tensorboard is True:
            tensorboard_dir = "{cfg.tensorboard_folder}/{cfg.experiment_name}/tensorboard/".format(cfg=cfg)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tensorboard_dir)

        # train
        for iteration in range(cfg.max_iter):
            images, labels = train_iterator.get()
            images, labels = images.cuda(), labels.cuda()
            masks_pred, iou_pred = self.model(images)
            total_loss = None
            loss_dict = {}
            for index, item in enumerate(self.losses.items()):
                tmp_loss = item[1](masks_pred, labels)
                loss_dict[item[0]] = tmp_loss.item()
                total_loss += cfg.loss_weight[index] * tmp_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iteration + 1) % cfg.log_iter == 0:
                write_log(log_path=log_path, log_data=train_meter.get(clear=True), status=self.exist_status[0],
                          writer=writer)
            # eval
            if (iteration + 1) % cfg.eval_iter == 0:
                mIoU, _ = self._eval()
                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    save_model(self.model, model_path)
                    print_and_save_log("saved model in {model_path}".format(model_path=model_path))
                    log_data = {'mIoU': mIoU, 'best_valid_mIoU': best_valid_mIoU}
                    write_log(log_path=log_path, log_data=log_data, status=self.exist_status[1], writer=writer)
        # final process
        save_model(self.model, model_path, is_final=True)
        if writer is not None:
            writer.close()

    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                masks_pred, iou_pred = self.model(images)
                predictions = torch.argmax(masks_pred, dim=1)
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    eval_metric.add(pred_mask, gt_mask)
        self.model.train()
        return eval_metric.get(clear=True)
