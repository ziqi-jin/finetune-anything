# @copyright ziqi-jin

import torch.nn as nn
import torch
from .segment_anything_ori.modeling.sam import Sam
from .utils import fix_params
from .segment_anything_ori.modeling.mask_decoder import MaskDecoder
from typing import List, Tuple
from torch.nn import functional as F
from .mask_decoder_heads import SemSegHead
from .mask_decoder_neck import MaskDecoderNeck


class BaseMaskDecoderAdapter(MaskDecoder):
    # is fix and load params
    def __init__(self, ori_sam: Sam, fix=False):
        super(BaseMaskDecoderAdapter, self).__init__()
        self.ori_sam_mask_decoder = ori_sam.mask_decoder
        if fix:
            fix_params(self.ori_sam_mask_decoder)  # move to runner to implement

    def forward(self, x):
        x = self.ori_sam_mask_decoder(x)
        return x


class SemMaskDecoderAdapter(BaseMaskDecoderAdapter):
    def __init__(self, ori_sam: Sam, fix=False):
        super.__init__(ori_sam, fix)
        self.decoder_neck = MaskDecoderNeck(transformer_dim=self.ori_sam_mask_decoder.transformer_dim,
                                            transformer=self.ori_sam_mask_decoder.transformer,
                                            num_multimask_outputs=self.ori_sam_mask_decoder.num_multimask_outputs)
        self.decoder_head = SemSegHead(transformer_dim=self.ori_sam_mask_decoder.transformer_dim,
                                       num_multimask_outputs=self.ori_sam_mask_decoder.num_multimask_outputs,
                                       iou_head_depth=self.ori_sam_mask_decoder.iou_head_depth,
                                       iou_head_hidden_dim=self.ori_sam_mask_decoder.iou_head_hidden_dim)
        # pair the params between ori mask_decoder and new mask_decoder_adapter
        self.pair_params(self.decoder_neck)
        self.pair_params(self.decoder_head)

    def forward(self, x, scale=1):
        masks, iou_pred = self.decoder_head(self.decoder_neck(x), scale=scale)
        return masks, iou_pred

    def pair_params(self, target_model: nn.Module):
        src_dict = self.ori_sam_mask_decoder.state_dict()
        for name, value in target_model.named_parameters():
            if name in src_dict:
                value.data.copy_(src_dict[name].data)

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
