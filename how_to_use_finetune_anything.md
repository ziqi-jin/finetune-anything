# How to use finetune-anything
finetune-anything (FA) is intended as a tool to help users quickly build extended SAM models. It not only supports the built-in basic tasks and basic models, but also supports user-defined extensions of different modules, training processes, and datasets for the extend SAM.


## Structure
Using FA can be divided into two parts: training and testing. The training part includes [model](#Model), [Datasets](#Datasets-and-Dataloader), [Losses](#Losses), [Optimizer](#Optimizer), [Logger](#Logger), and [Others](#Others).
The above content needs to be configured through the yaml file in `config`. The tasks already supported by FA can be trained and tested directly by inputting `task_name`.
```
CUDA_VISIBLE_DEVICES=${your GPU number} python train.py --task_name ${one of supported task names}
```
Custom configuration files can be trained and tested by reading `cfg`
```
CUDA_VISIBLE_DEVICES=${your GPU number} python train.py --cfg config/${yaml file name}
```
## Model
The SAM model includes image encdoer, prompt encoder and mask decoder. FA further encapsulates the encoder and decoder of SAM and identify Extend SAM model consists of image encoder adapter, prompt encdoer adapter and mask decoder adapter.
Users can choose the adapter that need to be fixed or learned during the finetune process. This function can be configured in the `model` part of the yaml file, as shown in the following example:

```yaml
model:
sam_name: 'extend sam name' # e.g., 'sem_sam', custom SAM model name, you should implement this model('sem_sam') first
params:
  # Fix the a part of parameters in SAM
  fix_img_en: True  # fix image encoder adapter parameters
  fix_prompt_en: True # fix prompt encoder adapter parameters
  fix_mask_de: False # unfix mask decoder adapter parameters to learn
  ckpt_path: 'your original sam weights'  # e.g., 'sam_ckpt/sam_vit_b_01ec64.pth' 
  class_num: 21 # number of classes for your dataset(20) + background(1)
  model_type: 'vit_b'    # type should be in [vit_h, vit_b, vit_l, default], this is original SAM type 
                         # related to different original SAM model. the type should be corresponded to the ckpt_path
```
### Customized Model

## Datasets and Dataloader

## Losses

## Optimizer

## Logger

## Others
