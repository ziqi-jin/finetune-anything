# copyright ziqi-jin
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from .image_encoder_adapter import BaseImgEncodeAdapter
from .mask_decoder_adapter import BaseMaskDecodeAdapter
from .prompt_encoder_adapter import BasePromptEncodeAdapter


class BaseExtendSam(nn.module):

    def __init__(self, ckpt_path=None, fix_img_en=False, fix_promt_en=False, fix_mask_de=False):
        ori_sam = sam_model_registry['default'](ckpt_path)
        self.img_adapter = BaseImgEncodeAdapter(ori_sam, fix=fix_img_en)
        self.prompt_adapter = BasePromptEncodeAdapter(ori_sam, fix=fix_promt_en)
        self.mask_adapter = BaseMaskDecodeAdapter(ori_sam, fix=fix_mask_de)

    def forword(self, img):
        x = self.img_adapter(img)
        prompt_embedding = self.prompt_adapter()
        outputs = self.mask_adapter(x, prompt_embedding)
        return outputs
