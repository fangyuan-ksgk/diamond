from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int # num_steps_conditioning = t
    cond_channels: int # cond_channels = e * t 
    depths: List[int] # ? what's the difference between this and the attn_depths?
    channels: List[int] # ? I guess we use convolution here ... ?
    attn_depths: List[bool] # ? what's the difference between this and depths?
    num_actions: Optional[int] = None # number of possible actions (vocabulary size)


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        
        # Embedding for action sequence: (batch, num_steps_conditioning) -> (batch, num_steps_conditioning * embedding_dim)
        embedding_dim = cfg.cond_channels // cfg.num_steps_conditioning
        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, embedding_dim),
            nn.Flatten(start_dim=1),  # (b, t, e) -> (b, t*e)
        )
        
        # how is this not redundant ... ?
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        
        # This guy mixes 'noisy next frame' with 'conditional frames' :: it only handles visual signals
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act)) # (b, t*e) -> (b, t*e) | Adding noise to the action? Augmentation of sorts?
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1)) # (b, (t+1)*c) -> (b, c) | We don't use transformer here? DiT, no? 
        x, _, _ = self.unet(x, cond) # (b, c) -> (b, c)
        x = self.conv_out(F.silu(self.norm_out(x))) # (b, c) -> (b, c)
        return x
