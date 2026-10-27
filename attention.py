import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

'''
    do the attention computation, including rope
    input params:
        x: [B, T, D]
    intermediate params:
        W_qkv: [D, 3D]
    output params:
        y: [B, T, D]
'''

class Attention(nn.Module):

    def __init__(self, T, D):
        super().__init__()

        self.input_shape = D
        self.output_shape = 3 * D
        self.num_head = 8
        self.head_dim = D // self.num_head
        if D % self.num_head != 0:
            raise ValueError(f"D={D} 必须能被 num_head={self.num_head} 整除")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} 必须能被 2 整除，用于 RoPe")
        
        # self.W_qkv = nn.Parameter(torch.randn([self.input_shape, self.output_shape]))
        self.W_qkv = nn.Linear(self.input_shape, self.output_shape)

        # RoPE
        self.pos_vec = torch.arange(0, T, 1).unsqueeze(-1)
        i = torch.arange(self.head_dim // 2, dtype=torch.float32)
        self.freq_vec = (10000.0 ** (-2 * i / self.head_dim)).unsqueeze(0)
        self.register_buffer("theta", self.pos_vec * self.freq_vec)   # [T, d/2]

        # Mask Attn
        self.W_o = nn.Linear(self.input_shape, self.input_shape)

    def RoPE(self, qk):

        # slice of q -> [B, nh, T, d/2]
        q_a = qk[:, :, :, 0:self.head_dim:2]
        q_b = qk[:, :, :, 1:self.head_dim:2]

        a = q_a * torch.cos(self.theta) - q_b * torch.sin(self.theta)
        b = q_a * torch.sin(self.theta) + q_b * torch.cos(self.theta)

        qk = torch.stack((a, b), dim=-1) # [B, nh, T, d//2, 2]
        qk = qk.flatten(-2)

        return qk

    def MaskAttention(self, q, k, v):

        s = q @ k.transpose(-1, -2) / sqrt(self.head_dim)

        T = q.shape[-2]
        mask = torch.ones(T, T, dtype=torch.bool, device=s.device).triu(diagonal=1)
        s = s.masked_fill(mask, float('-inf'))

        p = torch.softmax(s, dim=-1)
        o = p @ v   # [B, nd, T, d]

        # -> [B, T, nd, d] -> [B, T, D]
        o = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1)

        o = self.W_o(o)

        return o 

    def forward(self, x):

        x = self.W_qkv(x)
        q, k, v = x.chunk(3, dim=-1)

        # [B, T, D] -> [B, T, nh, d] -> [B, nh, T, d]
        q, k, v = map(
            lambda x: x.reshape(x.shape[0], x.shape[1], self.num_head, self.head_dim)
            .transpose(1, 2),
            (q, k, v)
        )

        # apply RoPE
        q, k = map(self.RoPE, (q, k))

        # apply Mask MHA
        o = self.MaskAttention(q, k, v)
        x = o

        return x