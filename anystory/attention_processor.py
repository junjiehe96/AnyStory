# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional, Dict, Any

import torch
from diffusers.models.attention_processor import Attention, F
from diffusers.models.embeddings import apply_rotary_emb
from torch import nn

from .lora_controller import enable_lora


def separable_scaled_dot_product_attention(query, key, value, attention_mask, model_config, routing_map=None):
    text_end = model_config["txt_seq_len"]
    text_redux_end = text_end + model_config["redux_seq_len"]
    text_redux_image_end = text_redux_end + model_config["img_seq_len"]
    text_redux_image_ref_end = text_redux_image_end + model_config["ref_seq_len"]
    text_redux_image_ref_router_end = text_redux_image_ref_end + model_config["router_seq_len"]

    # prepare attention mask if necessary
    ref_shift = model_config.get("ref_shift", 0.0)
    ref_mask = model_config.get("ref_mask", None)  # b, ref_seq_len, 1

    if ref_shift != 0 or ref_mask is not None or routing_map is not None:
        attention_mask = torch.zeros(
            query.shape[0], query.shape[1], query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
        )  # bs, heads, q_len, k_len

        if ref_shift != 0:
            # not shift ref self-attention
            attention_mask[:, :, :text_redux_image_end, text_redux_image_end:text_redux_image_ref_end] += ref_shift
        if ref_mask is not None and text_redux_image_ref_end > text_redux_image_end:
            ref_mask = ref_mask.transpose(-1, -2).unsqueeze(1)  # b, 1, 1, ref_seq_len
            ref_mask = (ref_mask - 1.) * 100.
            # not mask ref self-attention
            attention_mask[:, :, :text_redux_image_end, text_redux_image_end:text_redux_image_ref_end] += ref_mask

        if routing_map is not None:
            repeat_times = model_config["redux_seq_len"] // model_config["num_conds"]  # 81
            redux_routing_map = routing_map.unsqueeze(2).repeat((1, 1, repeat_times, 1)).flatten(1,
                                                                                                 2)  # bs, num*81, s*s
            redux_routing_map = (redux_routing_map.unsqueeze(1) - 1) * 100.  # bs, 1, num*81, s*s
            redux_routing_map = redux_routing_map.transpose(2, 3)  # bs, 1, s*s, num*81
            attention_mask[:, :, text_redux_end:text_redux_image_end, text_end:text_redux_end] += redux_routing_map

            repeat_times = model_config["ref_seq_len"] // model_config["num_conds"]  # 1024
            ref_routing_map = routing_map.unsqueeze(2).repeat((1, 1, repeat_times, 1)).flatten(1,
                                                                                               2)  # bs, num*1024, s*s
            ref_routing_map = (ref_routing_map.unsqueeze(1) - 1) * 100.  # bs, 1, num*1024, s*s
            ref_routing_map = ref_routing_map.transpose(2, 3)  # bs, 1, s*s, num*1024
            attention_mask[:, :, text_redux_end:text_redux_image_end,
            text_redux_image_end:text_redux_image_ref_end] += ref_routing_map

    hidden_states = F.scaled_dot_product_attention(
        query[:, :, :text_redux_image_end],
        key[:, :, :text_redux_image_ref_end], value[:, :, :text_redux_image_ref_end],
        attn_mask=(
            attention_mask[:, :, :text_redux_image_end,
            :text_redux_image_ref_end] if attention_mask is not None else None
        )
    )

    if text_redux_image_ref_end > text_redux_image_end and not model_config.get("use_ref_cache", False):
        # need to calculate ref_hidden_states additionally
        if model_config["num_conds"] == 1:
            ref_hidden_states = F.scaled_dot_product_attention(
                query[:, :, text_redux_image_end:text_redux_image_ref_end],
                key[:, :, text_redux_image_end:text_redux_image_ref_end],
                value[:, :, text_redux_image_end:text_redux_image_ref_end],
                attn_mask=None
            )
        else:
            ref_query = query[:, :, text_redux_image_end:text_redux_image_ref_end].unflatten(2, sizes=(
                model_config["num_conds"], -1))
            ref_key = key[:, :, text_redux_image_end:text_redux_image_ref_end].unflatten(2, sizes=(
                model_config["num_conds"], -1))
            ref_value = value[:, :, text_redux_image_end:text_redux_image_ref_end].unflatten(2, sizes=(
                model_config["num_conds"], -1))

            ref_hidden_states = F.scaled_dot_product_attention(ref_query, ref_key, ref_value, attn_mask=None)
            ref_hidden_states = ref_hidden_states.flatten(2, 3)

        hidden_states = torch.cat([hidden_states, ref_hidden_states], dim=2)

    if text_redux_image_ref_router_end > text_redux_image_ref_end:
        # need to calculate router_hidden_states additionally
        router_query = query[:, :, -model_config["router_seq_len"]:]
        router_key = key[:, :, -model_config["router_seq_len"]:]
        router_value = value[:, :, -model_config["router_seq_len"]:]

        router_key = torch.cat([key[:, :, text_redux_end:text_redux_image_end], router_key], dim=2)
        router_value = torch.cat([value[:, :, text_redux_end:text_redux_image_end], router_value], dim=2)

        router_hidden_states = F.scaled_dot_product_attention(router_query, router_key, router_value, attn_mask=None)

        hidden_states = torch.cat([hidden_states, router_hidden_states], dim=2)

    return hidden_states


class AnyStoryFluxAttnProcessor2_0(nn.Module):

    def __init__(self, hidden_size=3072, router_lora_rank=128, router_lora_bias=True):
        super().__init__()

        self.to_q_routing_map_lora_A = nn.Linear(hidden_size, router_lora_rank, bias=False)
        self.to_q_routing_map_lora_B = nn.Linear(router_lora_rank, hidden_size, bias=router_lora_bias)
        self.to_k_routing_map_lora_A = nn.Linear(hidden_size, router_lora_rank, bias=False)
        self.to_k_routing_map_lora_B = nn.Linear(router_lora_rank, hidden_size, bias=router_lora_bias)

        self.hidden_size = hidden_size
        self.ref_bank = {}

    def calculate_routing_maps(self, attn, encoder_hidden_states, hidden_states, router_hidden_states,
                               image_rotary_emb, router_rotary_emb, model_config):

        # here should output a routing map
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        head_dim = self.hidden_size // attn.heads

        if encoder_hidden_states is not None:
            # FluxTransformerBlock
            router_query_out = attn.add_q_proj(router_hidden_states) + \
                               self.to_q_routing_map_lora_B(self.to_q_routing_map_lora_A(router_hidden_states))
            router_query_out = router_query_out.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            router_query_out = attn.norm_added_q(router_query_out)
        else:
            # FluxSingleTransformerBlock
            router_query_out = attn.to_q(router_hidden_states) + \
                               self.to_q_routing_map_lora_B(self.to_q_routing_map_lora_A(router_hidden_states))
            router_query_out = router_query_out.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            router_query_out = attn.norm_q(router_query_out)
        router_query_out = apply_rotary_emb(router_query_out, router_rotary_emb)  # bs, h, num*9, d
        router_query_out = router_query_out.view(
            batch_size, attn.heads, model_config["num_conds"], -1, head_dim).mean((1, 3))  # bs, num, d

        if encoder_hidden_states is not None:
            # FluxTransformerBlock
            image_hidden_states = hidden_states
        else:
            # FluxSingleTransformerBlock
            image_hidden_states = hidden_states[:, -model_config["img_seq_len"]:]

        # `sample` projections.
        router_key_out = attn.to_k(image_hidden_states) + \
                         self.to_k_routing_map_lora_B(self.to_k_routing_map_lora_A(image_hidden_states))
        router_key_out = router_key_out.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        router_key_out = attn.norm_k(router_key_out)
        _image_rotary_emb = (
            image_rotary_emb[0][-model_config["img_seq_len"]:], image_rotary_emb[1][-model_config["img_seq_len"]:]
        )
        router_key_out = apply_rotary_emb(router_key_out, _image_rotary_emb)  # bs, h, s*s, d
        router_key_out = router_key_out.mean(1)  # bs, s*s, d

        router_logits = torch.bmm(router_query_out, router_key_out.transpose(-1, -2))  # bs, num, s*s
        index = router_logits.max(dim=1, keepdim=True)[1]  # bs, 1, s*s
        routing_map = torch.zeros_like(router_logits).scatter_(1, index, 1.0)  # bs, num, s*s

        return routing_map

    def forward(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor = None,
            encoder_hidden_states: torch.FloatTensor = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            ref_hidden_states: torch.FloatTensor = None,
            ref_rotary_emb: Optional[torch.Tensor] = None,
            router_hidden_states: torch.FloatTensor = None,
            router_rotary_emb: Optional[torch.Tensor] = None,
            model_config: Optional[Dict[str, Any]] = {},
    ):

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if ref_hidden_states is not None:
            with enable_lora((attn.to_q, attn.to_k, attn.to_v), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
                ref_query = attn.to_q(ref_hidden_states)
                ref_key = attn.to_k(ref_hidden_states)
                ref_value = attn.to_v(ref_hidden_states)

            ref_query = ref_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_key = ref_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_value = ref_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                ref_query = attn.norm_q(ref_query)
            if attn.norm_k is not None:
                ref_key = attn.norm_k(ref_key)

            if ref_rotary_emb is not None:
                ref_query = apply_rotary_emb(ref_query, ref_rotary_emb)
                ref_key = apply_rotary_emb(ref_key, ref_rotary_emb)

            query = torch.cat([query, ref_query], dim=2)
            key = torch.cat([key, ref_key], dim=2)
            value = torch.cat([value, ref_value], dim=2)

            if model_config.get("cache_ref", False):
                # cache ref k-v
                self.ref_bank["ref_key"] = ref_key
                self.ref_bank["ref_value"] = ref_value

        elif model_config.get("use_ref_cache", False):
            # fetch ref cache
            ref_key = self.ref_bank["ref_key"]
            ref_value = self.ref_bank["ref_value"]

            key = torch.cat([key, ref_key], dim=2)
            value = torch.cat([value, ref_value], dim=2)

        if router_hidden_states is not None:
            routing_map = self.calculate_routing_maps(
                attn, encoder_hidden_states, hidden_states, router_hidden_states, image_rotary_emb,
                router_rotary_emb, model_config
            )

            if encoder_hidden_states is not None:
                # FluxTransformerBlock
                with enable_lora((attn.add_q_proj, attn.add_k_proj, attn.add_v_proj), ("router_lora",),
                                 model_config.get("router_lora_scale", 1.0)):
                    router_query = attn.add_q_proj(router_hidden_states)
                    router_key = attn.add_k_proj(router_hidden_states)
                    router_value = attn.add_v_proj(router_hidden_states)

                router_query = router_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                router_key = router_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                router_value = router_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_added_q is not None:
                    router_query = attn.norm_added_q(router_query)
                if attn.norm_added_k is not None:
                    router_key = attn.norm_added_k(router_key)
            else:
                # FluxSingleTransformerBlock
                with enable_lora((attn.to_q, attn.to_k, attn.to_v), ("router_lora",),
                                 model_config.get("router_lora_scale", 1.0)):
                    router_query = attn.to_q(router_hidden_states)
                    router_key = attn.to_k(router_hidden_states)
                    router_value = attn.to_v(router_hidden_states)

                router_query = router_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                router_key = router_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                router_value = router_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    router_query = attn.norm_q(router_query)
                if attn.norm_k is not None:
                    router_key = attn.norm_k(router_key)

            if router_rotary_emb is not None:
                router_query = apply_rotary_emb(router_query, router_rotary_emb)
                router_key = apply_rotary_emb(router_key, router_rotary_emb)

            query = torch.cat([query, router_query], dim=2)
            key = torch.cat([key, router_key], dim=2)
            value = torch.cat([value, router_value], dim=2)
        else:
            routing_map = None

        # hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        hidden_states = separable_scaled_dot_product_attention(query, key, value, attention_mask,
                                                               model_config, routing_map)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        text_end = model_config["txt_seq_len"]
        text_redux_end = text_end + model_config["redux_seq_len"]
        text_redux_image_end = text_redux_end + model_config["img_seq_len"]
        text_redux_image_ref_end = text_redux_image_end + (
            model_config["ref_seq_len"] if not model_config.get("use_ref_cache", False) else 0)
        text_redux_image_ref_router_end = text_redux_image_ref_end + model_config["router_seq_len"]

        assert hidden_states.shape[1] == text_redux_image_ref_router_end

        if router_hidden_states is not None:
            router_hidden_states = hidden_states[:, text_redux_image_ref_end: text_redux_image_ref_router_end]
        if ref_hidden_states is not None:
            ref_hidden_states = hidden_states[:, text_redux_image_end: text_redux_image_ref_end]

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : text_redux_end],
                hidden_states[:, text_redux_end: text_redux_image_end],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if ref_hidden_states is not None:
                with enable_lora((attn.to_out[0],), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
                    ref_hidden_states = attn.to_out[0](ref_hidden_states)
                    ref_hidden_states = attn.to_out[1](ref_hidden_states)

            if router_hidden_states is not None:
                with enable_lora((attn.to_add_out,), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
                    router_hidden_states = attn.to_add_out(router_hidden_states)

            return hidden_states, encoder_hidden_states, ref_hidden_states, router_hidden_states

        else:
            hidden_states = hidden_states[:, : text_redux_image_end]

            return hidden_states, ref_hidden_states, router_hidden_states
