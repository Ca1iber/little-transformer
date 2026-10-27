import torch
import torch.nn as nn


'''
    implement the feed forward network thing
    using silu function
    input params: 
        x: [B, T, D]
    intermediate params:
        W_up: [D, 3*D]
        W_gate: [D, 3*D]
        W_down: [3*D, D]
    output params:
        y: [B, T, D]
'''

class FFN(nn.Module):

    def __init__(self, D):
        super().__init__()

        self.W_up_gate = nn.Linear(D, 6 * D)
        self.W_down = nn.Linear(3 * D, D)


    def forward(self, x):

        # [D, 6*D]
        x = self.W_up_gate(x)

        # [D, 3*D]
        gate, up = x.chunk(2, dim=-1)

        gate = gate * torch.sigmoid(gate)

        intermediate = gate * up

        x = self.W_down(intermediate)

        return x