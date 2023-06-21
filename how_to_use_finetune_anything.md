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

## Datasets and Dataloader

## Losses

## Optimizer

## Logger

## Others
