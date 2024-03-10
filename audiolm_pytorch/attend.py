import torch
from torch import nn, einsum
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# constants

Config = namedtuple('Config', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        k = repeat(k, 'b ... -> b h ...', h = heads)
        v = repeat(v, 'b ... -> b h ...', h = heads)

        causal = self.causal

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

            if causal:
                causal_mask = torch.ones((q_len, k_len), device = q.device, dtype = torch.bool).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask                
                causal = False

        config = self.cuda_config if is_cuda else self.cpu_config
            
        return flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0., causal=causal)

    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5
        
        return self.flash_attn(q, k, v, mask = mask)