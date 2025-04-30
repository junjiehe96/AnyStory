# Modified from https://github.com/Yuanshi9815/OminiControl/blob/main/src/flux/lora_controller.py
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Optional, Type, Sequence

from peft.tuners.lora import LoraLayer


class enable_lora:
    def __init__(self, lora_modules: Sequence[LoraLayer], adapters: Sequence[str], scales: Sequence[float] = 1.):
        if not isinstance(adapters, Sequence):
            adapters = [adapters]
        if not isinstance(scales, Sequence):
            scales = [scales] * len(adapters)
        assert len(adapters) == len(scales), "the number of adapters does not match the number of scales."

        self.lora_modules = [each for each in lora_modules if isinstance(each, LoraLayer)]
        self.adapters = list(adapters)
        self.scales = list(scales)

        self.original_scales = [
            {
                adapter: lora_module.scaling.get(adapter, 0.0)
                for adapter in adapters
            }
            for lora_module in self.lora_modules
        ]
        self.original_active_adapters = [
            lora_module.active_adapters
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            lora_module._active_adapter = list(set(self.original_active_adapters[i] + self.adapters))
            for adapter, scale in zip(self.adapters, self.scales):
                lora_module.set_scale(adapter, scale)

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            lora_module._active_adapter = list(self.original_active_adapters[i])
            for adapter in self.adapters:
                lora_module.set_scale(adapter, self.original_scales[i][adapter])
                assert self.original_scales[i][adapter] == 0.0
