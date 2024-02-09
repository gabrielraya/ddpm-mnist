"""
Various utilities for neural networks.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, frequency_embedding_size=256, hidden_size=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Positional Encoding is perform as follows:

            PE(pos,2i) = sin(pos/100002i/d_model )
            PE(pos,2i+1) = cos(pos/100002i/dmodel )

    :param timesteps: a 1-D Tensor of N indices, one per batch element. So one t per sample x -> x(t)
                      These may be fractional
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings. magic number 10000 is from transformers
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(- math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)/half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Patch(nn.Module):
    """ Transform input tensor into a sequence of flatten patches or revert it back to original shape
    Args:
        input_size (int): Expected dimension (height or width) of the input tensor. Default is 32
        patch_size (int): Size of each square patch. Default is 4, means a patch of 4x4
    Returns:
        patches (torch.tensor): Output tensor with dimensions [..., (h/dim) * (w/dim), c * dim * dim],
    where 'h' and 'w' are the height and width of the input tensor, and 'c' is the number of channels.

    The 'reverse' flag in the forward method determines whether to split the tensor into patches
    or to revert the patches back to the original tensor shape.
    """

    def __init__(self, input_size: int = 32, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        self.r = input_size // patch_size
        self.num_patches = self.r ** 2    # Number of patches

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            return rearrange(x, "... (h w) (c p1 p2) -> ... c (h p1) (w p2)", h=self.r, w=self.r,
                             p1=self.patch_size, p2=self.patch_size)

        # Validate input dimensions
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            raise ValueError("Input tensor dimensions do not match expected dimensions.")

        return rearrange(x, "... c (h p1) (w p2) -> ... (h w) (c p1 p2)", p1=self.patch_size,  p2=self.patch_size)



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

