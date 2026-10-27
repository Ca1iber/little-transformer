import torch
import torch.nn as nn
import torch.nn.functional as F


'''
    normalize the input to make the value stable
    input shape: [B, T, D]
    output shape: [B, T, D]
    input params: 
        x: [B, T, D]
    intermediate params: 
        gamma: [1, 1, D]
        beta: [1, 1, D]
        average: [B, T, 1]
        sigma: [B, T, 1]
    output params: 
        y: [B, T, D]

'''

class LayerNorm(nn.Module):

    def __init__(self, D, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(D))
        self.beta = nn.Parameter(torch.zeros(D))

    def forward(self, x):

        average = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + self.epsilon)
        x = (x - average) / sigma
        x = x * self.gamma + self.beta
        return x

        '''
        return F.layer_norm(
            x,
            normalized_shape=self.gamma.shape,
            weight=self.gamma,
            bias=self.beta,
            eps=self.epsilon,
        )
        '''



