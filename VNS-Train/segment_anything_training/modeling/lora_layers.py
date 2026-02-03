import torch
from torch import nn
from torch.nn import functional as F
from typing import Type

class LoRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, r=4, scale=1) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.r = r
        self.trainable_lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.trainable_lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.trainable_lora_down.weight, std=1/r)
        nn.init.zeros_(self.trainable_lora_up.weight)

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias) + self.scale*self.dropout(self.trainable_lora_up(self.selector(self.trainable_lora_down(input))))
        return  out

class LoRAConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode: str = 'zeros', device=None, dtype=None, r=4, scale=1) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        assert type(kernel_size) is int
        self.r = r
        self.scale = scale
        
        self.trainable_lora_down = nn.Conv2d(
            in_channels = in_channels,
            out_channels = r,
            kernel_size = kernel_size,
            bias=False
        )

        self.dropout = nn.Dropout(0.1)

        self.trainable_lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.trainable_lora_down.weight, std=1/r)
        nn.init.zeros_(self.trainable_lora_up.weight)

    def forward(self, input):
        out = F.conv2d(input, self.weight, self.bias, self.stride)
        out = out + self.scale*self.dropout(self.trainable_lora_up(self.selector(self.trainable_lora_down(input))))
        return out


