# -*- coding:utf-8 -*-
"""
authorï¼šSylvan Peng
dateï¼?024å¹?3æœ?2æ—?
versionï¼?.0

#è¿˜æœªå®Œæˆ
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

