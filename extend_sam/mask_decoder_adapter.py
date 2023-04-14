# copyright ziqi-jin

import torch.nn as nn
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
class BaseMaskDecodeAdapter(nn.Module):

    def __init__(self, ori_sam:Sam, fix=False):
        self.ori_sam_mask_decoder = ori_sam.mask_decoder
        if fix:
            fix_params(self.ori_sam_mask_decoder)

    def forward(self, x):
        x = self.ori_sam_mask_decoder(x)
        return x