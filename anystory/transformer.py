# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Dict, Any

import torch
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
)

from .block import block_forward, single_block_forward
from .lora_controller import enable_lora


def prepare_params(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
):
    return (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        return_dict,
    )


def tranformer_forward(
        transformer: FluxTransformer2DModel,
        redux_hidden_states: torch.Tensor = None,
        redux_ids: torch.Tensor = None,
        ref_hidden_states: torch.Tensor = None,
        ref_ids: torch.Tensor = None,
        ref_t: float = 0,
        router_hidden_states: torch.Tensor = None,
        router_ids: torch.Tensor = None,
        model_config: Optional[Dict[str, Any]] = {},
        **params,
):
    self = transformer

    (
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
        joint_attention_kwargs,
        return_dict,
    ) = prepare_params(**params)

    # required parameters calculated internally
    model_config["txt_seq_len"] = encoder_hidden_states.shape[1]
    model_config["img_seq_len"] = hidden_states.shape[1]
    model_config["redux_seq_len"] = redux_hidden_states.shape[1] if redux_hidden_states is not None else 0
    model_config["ref_seq_len"] = ref_hidden_states.shape[1] if ref_hidden_states is not None else 0
    model_config["router_seq_len"] = router_hidden_states.shape[1] if router_hidden_states is not None else 0

    use_ref_cond = ref_hidden_states is not None and not model_config.get("use_ref_cache", False)
    use_router = router_hidden_states is not None

    if redux_hidden_states is not None:
        encoder_hidden_states = torch.cat([encoder_hidden_states, redux_hidden_states], dim=1)
        txt_ids = torch.cat([txt_ids, redux_ids], dim=0)

    hidden_states = self.x_embedder(hidden_states)  # Nx1024x64 -> Nx1024x3072
    with enable_lora((self.x_embedder,), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
        ref_hidden_states = self.x_embedder(ref_hidden_states) if use_ref_cond else None

    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )

    ref_temb = (
        self.time_text_embed(torch.ones_like(timestep) * ref_t * 1000, pooled_projections)
        if guidance is None else
        self.time_text_embed(torch.ones_like(timestep) * ref_t * 1000, guidance, pooled_projections)
    ) if use_ref_cond else None

    router_temb = temb if use_router else None

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)  # Nx512x4096 -> Nx512x3072
    with enable_lora((self.context_embedder,), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
        router_hidden_states = self.context_embedder(router_hidden_states) if use_router else None

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    if use_ref_cond:
        ref_rotary_emb = self.pos_embed(ref_ids)
    if use_router:
        router_rotary_emb = self.pos_embed(router_ids)

    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs = {"use_reentrant": False}
            encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states = (
                torch.utils.checkpoint.checkpoint(
                    block_forward,
                    self=block,
                    model_config=model_config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    ref_hidden_states=ref_hidden_states if use_ref_cond else None,
                    ref_temb=ref_temb if use_ref_cond else None,
                    ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
                    router_hidden_states=router_hidden_states if use_router else None,
                    router_temb=router_temb if use_router else None,
                    router_rotary_emb=router_rotary_emb if use_router else None,
                    **ckpt_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states = block_forward(
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states if use_ref_cond else None,
                ref_temb=ref_temb if use_ref_cond else None,
                ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
                router_hidden_states=router_hidden_states if use_router else None,
                router_temb=router_temb if use_router else None,
                router_rotary_emb=router_rotary_emb if use_router else None,
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            ckpt_kwargs = {"use_reentrant": False}
            hidden_states, ref_hidden_states, router_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward,
                self=block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states if use_ref_cond else None,
                ref_temb=ref_temb if use_ref_cond else None,
                ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
                router_hidden_states=router_hidden_states if use_router else None,
                router_temb=router_temb if use_router else None,
                router_rotary_emb=router_rotary_emb if use_router else None,
                **ckpt_kwargs,
            )
        else:
            hidden_states, ref_hidden_states, router_hidden_states = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states if use_ref_cond else None,
                ref_temb=ref_temb if use_ref_cond else None,
                ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
                router_hidden_states=router_hidden_states if use_router else None,
                router_temb=router_temb if use_router else None,
                router_rotary_emb=router_rotary_emb if use_router else None,
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
