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
The SAM model includes image encdoer, prompt encoder and mask decoder. FA further encapsulates the encoder and decoder of SAM and identify Extend-SAM model consists of image encoder adapter, prompt encoder adapter and mask decoder adapter. The initialized process of Extend-SAM as below,
<img width="960" src="https://user-images.githubusercontent.com/67993288/248108534-62a4e5aa-cf4f-41f9-b745-db2924a376bc.svg">

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
If you need to redesign the structure of a certain module of SAM, you need to write code according to the following three steps. Take [SemanticSAM](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/extend_sam.py#L43) as an example.
- step1

First, inherit the corresponding adapter base class in `extend_sam\xxx_(encoder or decoder)_adapter.py`, and then implement the `__init__` and `forward` function corresponding to the adapter.
```python
class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False, class_num=20):
        super(SemMaskDecoderAdapter, self).__init__(ori_sam, fix) # init super class
        self.decoder_neck = MaskDecoderNeck(...) # custom module
        self.decoder_head = SemSegHead(...) # custom module
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck) # give the weights which are with the same name in original SAM to customized module
        self.pair_params(self.decoder_head)

    def forward(self, ...):
        ... = self.decoder_neck(...)
        masks, iou_pred = self.decoder_head(...)
        return masks, iou_pred
```
- step2

First inherit the BaseExtendSAM base class in [extend_sam.py](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/extend_sam.py#L43), and make necessary modifications to `__init__` function.
```python
class SemanticSam(BaseExtendSam):

    def __init__(self, ...):
        super().__init__(...) # init super class
        self.mask_adapter = SemMaskDecoderAdapter(...) # replace original Adapter as the  new identified customized Adapter 
```
- step3

Add new Extend-SAM class to [AVAI_MODEL](https://github.com/ziqi-jin/finetune-anything/blob/350c1fbf7f122a8525e7ffdecc40f259b262983f/extend_sam/__init__.py#L10) dict and give it a key.
then you can train this new model by modify the `sam_name` in config file.
## Datasets and Dataloader

## Losses

## Optimizer

## Logger

## Others
