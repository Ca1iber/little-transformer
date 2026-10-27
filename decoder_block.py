import torch
import torch.nn as nn

from attention import Attention
from norm import LayerNorm
from ffn import FFN

class DecoderBlock(nn.Module):

    def __init__(self, T, D):
        super().__init__()

        self.layer_norm_1 = LayerNorm(D, epsilon=1e-5)
        self.layer_norm_2 = LayerNorm(D, epsilon=1e-5)
        self.mask_attn = Attention(T, D)
        self.ffn = FFN(D)

    def forward(self, x):

        y = self.layer_norm_1(x)
        y = self.mask_attn(y)
        x = x + y

        y = self.layer_norm_2(x)
        y = self.ffn(y)
        x = x + y

        return x