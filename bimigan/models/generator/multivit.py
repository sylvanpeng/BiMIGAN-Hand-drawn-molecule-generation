# -*- coding:utf-8 -*-
"""
author：Sylvan Peng
date�?024�?3�?2�?
version�?.0

#还未完成
"""
import numpy as np
from torch import nn

from bimigan.torch.layers.transformer import (
    calc_tokenized_size, ViTInput, TransformerEncoder, img_to_tokens,
    img_from_tokens
)


class MultiModalViTUNetGenerator(nn.Module):
    def __init__(
            self, features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm, image_shape, token_size, rescale=False, rezero=True,
            **kwargs):
        super().__init__(**kwargs)
    def forward(self):
        None

