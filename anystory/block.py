# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Dict, Any

import torch

from .lora_controller import enable_lora


def block_forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        ref_hidden_states: torch.FloatTensor = None,
        ref_temb: torch.FloatTensor = None,
        ref_rotary_emb=None,
        router_hidden_states: torch.FloatTensor = None,
        router_temb: torch.FloatTensor = None,
        router_rotary_emb=None,
        model_config: Optional[Dict[str, Any]] = {},
):
    use_ref_cond = ref_hidden_states is not None
    use_router = router_hidden_states is not None

    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_ref_cond:
        with enable_lora((self.norm1.linear,), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
            (
                norm_ref_hidden_states,
                ref_gate_msa,
                ref_shift_mlp,
                ref_scale_mlp,
                ref_gate_mlp,
            ) = self.norm1(ref_hidden_states, emb=ref_temb)

    if use_router:
        with enable_lora((self.norm1_context.linear,), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
            (
                norm_router_hidden_states,
                router_gate_msa,
                router_shift_mlp,
                router_scale_mlp,
                router_gate_mlp,
            ) = self.norm1_context(router_hidden_states, emb=router_temb)

    # Attention.
    attn_output, context_attn_output, ref_attn_output, router_attn_output = self.attn.processor(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        ref_hidden_states=norm_ref_hidden_states if use_ref_cond else None,
        ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
        router_hidden_states=norm_router_hidden_states if use_router else None,
        router_rotary_emb=router_rotary_emb if use_router else None,
    )

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    # 3. ref_hidden_states
    if use_ref_cond:
        ref_attn_output = ref_gate_msa.unsqueeze(1) * ref_attn_output
        ref_hidden_states = ref_hidden_states + ref_attn_output

    # 4. router_hidden_states
    if use_router:
        router_attn_output = router_gate_msa.unsqueeze(1) * router_attn_output
        router_hidden_states = router_hidden_states + router_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    # 3. ref_hidden_states
    if use_ref_cond:
        norm_ref_hidden_states = self.norm2(ref_hidden_states)
        norm_ref_hidden_states = norm_ref_hidden_states * (1 + ref_scale_mlp[:, None]) + ref_shift_mlp[:, None]

    # 4. router_hidden_states
    if use_router:
        norm_router_hidden_states = self.norm2_context(router_hidden_states)
        norm_router_hidden_states = norm_router_hidden_states * (1 + router_scale_mlp[:, None]) + router_shift_mlp[:,
                                                                                                  None]

    # Feed-forward.
    # 1. hidden_states
    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output

    # 3. ref_hidden_states
    if use_ref_cond:
        with enable_lora((self.ff.net[2],), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
            ref_ff_output = self.ff(norm_ref_hidden_states)
            ref_ff_output = ref_gate_mlp.unsqueeze(1) * ref_ff_output

    # 4. router_hidden_states
    if use_router:
        with enable_lora((self.ff_context.net[2],), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
            router_ff_output = self.ff_context(norm_router_hidden_states)
            router_ff_output = router_gate_mlp.unsqueeze(1) * router_ff_output

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_ref_cond:
        ref_hidden_states = ref_hidden_states + ref_ff_output
    if use_router:
        router_hidden_states = router_hidden_states + router_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states


def single_block_forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        ref_hidden_states: torch.FloatTensor = None,
        ref_temb: torch.FloatTensor = None,
        ref_rotary_emb=None,
        router_hidden_states: torch.FloatTensor = None,
        router_temb: torch.FloatTensor = None,
        router_rotary_emb=None,
        model_config: Optional[Dict[str, Any]] = {},
):
    use_ref_cond = ref_hidden_states is not None
    use_router = router_hidden_states is not None

    residual = hidden_states

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    if use_ref_cond:
        residual_ref = ref_hidden_states
        with enable_lora((self.norm.linear, self.proj_mlp), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
            norm_ref_hidden_states, ref_gate = self.norm(ref_hidden_states, emb=ref_temb)
            mlp_ref_hidden_states = self.act_mlp(self.proj_mlp(norm_ref_hidden_states))

    if use_router:
        residual_router = router_hidden_states
        with enable_lora((self.norm.linear, self.proj_mlp), ("router_lora",),
                         model_config.get("router_lora_scale", 1.0)):
            norm_router_hidden_states, router_gate = self.norm(router_hidden_states, emb=router_temb)
            mlp_router_hidden_states = self.act_mlp(self.proj_mlp(norm_router_hidden_states))

    attn_output, ref_attn_output, router_attn_output = self.attn.processor(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        ref_hidden_states=norm_ref_hidden_states if use_ref_cond else None,
        ref_rotary_emb=ref_rotary_emb if use_ref_cond else None,
        router_hidden_states=norm_router_hidden_states if use_router else None,
        router_rotary_emb=router_rotary_emb if use_router else None,
    )

    # proj_out: 3072+3072*4(mlp) -> 3072
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states

    if use_ref_cond:
        with enable_lora((self.proj_out,), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
            ref_hidden_states = torch.cat([ref_attn_output, mlp_ref_hidden_states], dim=2)
            ref_gate = ref_gate.unsqueeze(1)
            ref_hidden_states = ref_gate * self.proj_out(ref_hidden_states)
            ref_hidden_states = residual_ref + ref_hidden_states

    if use_router:
        with enable_lora((self.proj_out,), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
            router_hidden_states = torch.cat([router_attn_output, mlp_router_hidden_states], dim=2)
            router_gate = router_gate.unsqueeze(1)
            router_hidden_states = router_gate * self.proj_out(router_hidden_states)
            router_hidden_states = residual_router + router_hidden_states

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states, ref_hidden_states, router_hidden_states
