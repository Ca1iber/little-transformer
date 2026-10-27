import torch
import torch.nn as nn

from decoder_block import DecoderBlock
from norm import LayerNorm

class CZWModel(nn.Module):

    def __init__(self, T, D, num_layer, vocab_size):
        super().__init__()

        self.num_layer = num_layer
        self.vocab_size = vocab_size
        self.decoder_blocks = nn.ModuleList(DecoderBlock(T, D) for i in range(self.num_layer))
        self.embedding = nn.Embedding(vocab_size, D)

        self.final_layer_norm = LayerNorm(D, epsilon=1e-5)
        self.lm_head = nn.Linear(D, self.vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        for i in range(self.num_layer):
            x = self.decoder_blocks[i](x)

        x = self.final_layer_norm(x)

        x = self.lm_head(x)

        return x