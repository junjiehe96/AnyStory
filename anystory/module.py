# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines import ReduxImageEncoder


class AnyStoryReduxImageEncoder(ReduxImageEncoder):

    def __init__(
            self,
            redux_dim: int = 1152,
            txt_in_features: int = 4096,
            lora_rank: int = 128,
            lora_bias: bool = True,
            output_size: int = 9,
    ) -> None:
        super().__init__(redux_dim=redux_dim, txt_in_features=txt_in_features)

        self.redux_up_lora_A = nn.Linear(redux_dim, lora_rank, bias=False)
        self.redux_up_lora_B = nn.Linear(lora_rank, txt_in_features * 3, bias=lora_bias)

        self.redux_down_lora_A = nn.Linear(txt_in_features * 3, lora_rank, bias=False)
        self.redux_down_lora_B = nn.Linear(lora_rank, txt_in_features, bias=lora_bias)

        self.gate_proj = nn.Linear(txt_in_features, 1)
        self.pooling_head = nn.AdaptiveAvgPool2d(output_size=(output_size, output_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x.shape: b, 729, 1152
        # mask.shape: b, 729, 1

        x = self.redux_up(x) + self.redux_up_lora_B(self.redux_up_lora_A(x))
        x = F.silu(x)
        x = self.redux_down(x) + self.redux_down_lora_B(self.redux_down_lora_A(x))  # b, 729, 4096

        gate = F.sigmoid(self.gate_proj(x))  # b, 729, 1
        x = x * gate  # b, 729, 4096

        b, l, d = x.shape
        s = int(l ** 0.5)
        x = x.transpose(1, 2).reshape(b, d, s, s)  # b, 4096, 27, 27
        x = self.pooling_head(x)  # b, 4096, 9, 9

        gate = gate.transpose(1, 2).reshape(b, -1, s, s)  # b, 1, 27, 27
        gate = self.pooling_head(gate)  # b, 1, 9, 9

        mask = mask.transpose(1, 2).reshape(b, -1, s, s)  # b, 1, 27, 27
        mask = (self.pooling_head(mask) > 0).to(x)  # b, 1, 9, 9

        x = (x / (gate + 1e-6)) * mask

        x = x.flatten(2).transpose(1, 2)  # b, 81, 4096

        return x
