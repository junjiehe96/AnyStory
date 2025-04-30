# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List, Union, Optional, Dict, Any, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput, calculate_shift, retrieve_timesteps
from transformers.models.siglip import SiglipImageProcessor, SiglipVisionModel

from .attention_processor import AnyStoryFluxAttnProcessor2_0
from .module import AnyStoryReduxImageEncoder
from .transformer import tranformer_forward


class AnyStoryFluxPipeline:

    def __init__(
            self,
            hf_flux_pipeline_path: str,
            hf_flux_redux_path: str,
            anystory_path: str,

            ref_size=512,
            device=torch.device("cuda"),
            torch_dtype=torch.bfloat16,
    ):
        flux_pipeline = self.init_flux_pipline(hf_flux_pipeline_path, device, torch_dtype)
        siglip_image_processor = SiglipImageProcessor(size={"height": 384, "width": 384})
        siglip_image_encoder = self.init_siglip_image_encoder(hf_flux_redux_path, device, torch_dtype)
        redux_embedder = self.init_redux_embedder(hf_flux_redux_path, device, torch_dtype)
        router_embedder = self.init_router_embedder(hf_flux_redux_path, device, torch_dtype)

        self.load_anystory_model(flux_pipeline, redux_embedder, router_embedder, anystory_path)

        self.flux_pipeline = flux_pipeline
        self.siglip_image_processor = siglip_image_processor
        self.siglip_image_encoder = siglip_image_encoder
        self.redux_embedder = redux_embedder
        self.router_embedder = router_embedder

        self.ref_size = ref_size
        self.device = device
        self.torch_dtype = torch_dtype

    @classmethod
    def init_flux_pipline(cls, hf_flux_pipeline_path, device, torch_dtype):
        flux_pipeline = FluxPipeline.from_pretrained(hf_flux_pipeline_path, torch_dtype=torch_dtype).to(device)

        attn_procs = {}
        for name in flux_pipeline.transformer.attn_processors.keys():
            if name.endswith("attn.processor"):
                attn_procs[name] = AnyStoryFluxAttnProcessor2_0(
                    hidden_size=(
                            flux_pipeline.transformer.config.attention_head_dim *
                            flux_pipeline.transformer.config.num_attention_heads
                    ),
                    router_lora_rank=128,
                    router_lora_bias=True,
                ).to(device=device, dtype=torch_dtype)
        flux_pipeline.transformer.set_attn_processor(attn_procs)

        return flux_pipeline

    @classmethod
    def init_siglip_image_encoder(cls, hf_flux_redux_path, device, torch_dtype):
        siglip_image_encoder = SiglipVisionModel.from_pretrained(
            hf_flux_redux_path,
            subfolder="image_encoder",
            torch_dtype=torch_dtype
        ).to(device=device)
        return siglip_image_encoder

    @classmethod
    def init_redux_embedder(cls, hf_flux_redux_path, device, torch_dtype):
        redux_embedder = AnyStoryReduxImageEncoder.from_pretrained(
            hf_flux_redux_path,
            subfolder="image_embedder",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            device_map=None,
            output_size=9,
            lora_rank=128,
            lora_bias=True
        ).to(device)
        return redux_embedder

    @classmethod
    def init_router_embedder(cls, hf_flux_redux_path, device, torch_dtype):
        router_embedder = AnyStoryReduxImageEncoder.from_pretrained(
            hf_flux_redux_path,
            subfolder="image_embedder",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            device_map=None,
            output_size=3,
        ).to(device)
        return router_embedder

    @classmethod
    def load_anystory_model(cls, flux_pipeline, redux_embedder, router_embedder, anystory_path):
        print(f"loading anystory model from {anystory_path}...")
        state_dict = torch.load(anystory_path, weights_only=True, map_location="cpu")

        # load ref
        flux_pipeline.load_lora_weights(state_dict["ref"], adapter_name="ref_lora")
        # load redux
        redux_embedder.load_state_dict(state_dict["redux"], strict=False)
        # load router
        router_embedder_sd = {}
        router_adapter_sd = {}
        router_lora_sd = {}
        for k, v in state_dict["router"].items():
            if k.startswith("router_embedder."):
                router_embedder_sd[k[len("router_embedder."):]] = v
            elif "attn.processor" in k:
                router_adapter_sd[k[len("transformer."):]] = v
            else:
                router_lora_sd[k] = v

        router_embedder.load_state_dict(router_embedder_sd, strict=False)
        flux_pipeline.transformer.load_state_dict(router_adapter_sd, strict=False)
        flux_pipeline.load_lora_weights(router_lora_sd, adapter_name="router_lora")

        # deactivate by default
        flux_pipeline.transformer.set_adapters(["ref_lora", "router_lora"], [0.0, 0.0])

    def encode_redux_maybe_with_router_condition(self, image, mask, enable_router=False):
        image = self.siglip_image_processor.preprocess(images=image.convert("RGB"), return_tensors="pt").pixel_values
        mask = self.siglip_image_processor.preprocess(images=mask.convert("RGB"), resample=0, do_normalize=False,
                                                      return_tensors="pt").pixel_values
        image = (image * mask).to(device=self.device, dtype=self.torch_dtype)
        mask = mask.mean(dim=1, keepdim=True).to(device=self.device, dtype=self.torch_dtype)

        # encoding
        siglip_output = self.siglip_image_encoder(image).last_hidden_state
        s = self.siglip_image_encoder.vision_model.embeddings.image_size \
            // self.siglip_image_encoder.vision_model.embeddings.patch_size
        mask = (F.adaptive_avg_pool2d(mask, output_size=(s, s)) > 0).to(dtype=self.torch_dtype)
        mask = mask.flatten(2).transpose(1, 2)  # bs, 729, 1
        redux_embeds = self.redux_embedder(siglip_output, mask)  # [bs, 81, 4096]
        redux_ids = torch.zeros((redux_embeds.shape[1], 3), device=self.device, dtype=self.torch_dtype)

        if enable_router:
            router_embeds = self.router_embedder(siglip_output, mask)  # [bs, 1, 4096]
            router_ids = torch.zeros((router_embeds.shape[1], 3), device=self.device, dtype=self.torch_dtype)
        else:
            router_embeds = router_ids = None

        return (redux_embeds, redux_ids), (router_embeds, router_ids)

    def encode_ref_condition(self, image, mask, position_delta=None):
        image = self.flux_pipeline.image_processor.preprocess(image.convert("RGB"),
                                                              height=self.ref_size, width=self.ref_size)
        mask = mask.convert("L").resize((self.ref_size, self.ref_size), resample=0)
        mask = torch.from_numpy(np.array(mask)).permute(0, 1).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]

        image = (image * mask).to(device=self.device, dtype=self.torch_dtype)
        mask = mask.to(device=self.device, dtype=self.torch_dtype)

        # encoding
        latent = self.flux_pipeline.vae.encode(image).latent_dist.sample()
        latent = (latent - self.flux_pipeline.vae.config.shift_factor) * self.flux_pipeline.vae.config.scaling_factor
        ref_embeds = self.flux_pipeline._pack_latents(latent, *latent.shape)
        ref_ids = self.flux_pipeline._prepare_latent_image_ids(
            latent.shape[0],
            latent.shape[2] // 2,
            latent.shape[3] // 2,
            self.flux_pipeline.device,
            self.flux_pipeline.dtype,
        )

        if position_delta is None:
            position_delta = [0, -self.ref_size // 16]  # width shift
        ref_ids[:, 1] += position_delta[0]
        ref_ids[:, 2] += position_delta[1]

        s = self.ref_size // 16
        ref_masks = (F.adaptive_avg_pool2d(mask, output_size=(s, s)) > 0).to(self.device, dtype=self.torch_dtype)
        ref_masks = ref_masks.flatten(2).transpose(1, 2)  # bs, 1024, 1

        return (ref_embeds, ref_ids), ref_masks

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            images: List[Image.Image] = None,
            masks: List[Image.Image] = None,
            seed: Optional[int] = None,
            enable_router: bool = False,
            redux_scale=0.6,
            redux_start_at=0.1,
            redux_end_at=0.3,
            ref_shift=0.0,
            enable_ref_mask=True,
            ref_start_at=0.0,
            ref_end_at=1.0,
            enable_ref_cache=True,
            **params,
    ):
        if seed is None or seed == -1:
            generator = None
        else:
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
            np.random.seed(seed)
            generator = torch.Generator(self.device).manual_seed(seed)

        redux_conditions = []
        ref_conditions = []
        router_conditions = []
        ref_mask = None
        if images is not None:
            for image, mask in zip(images, masks):
                redux_with_router_condition = self.encode_redux_maybe_with_router_condition(image, mask, enable_router)
                redux_conditions.append(redux_with_router_condition[0])
                if enable_router:
                    router_conditions.append(redux_with_router_condition[1])

                ref_condition, ref_masks = self.encode_ref_condition(image, mask)
                ref_conditions.append(ref_condition)
                if enable_ref_mask:
                    # TODO: remove ref mask
                    # Here, we use ref_mask instead of background routing to alleviate the impact of inherent biases in
                    # reference background features (learned from limited training data) on the quality of generated
                    # images.
                    if ref_mask is None:
                        ref_mask = ref_masks
                    else:
                        ref_mask = torch.cat([ref_mask, ref_masks], dim=1)  # b, 1024, 1

        prompt_embeds, pooled_prompt_embeds, _ = self.flux_pipeline.encode_prompt(prompt, prompt_2=None)

        images = generate(
            self.flux_pipeline,
            redux_conditions=redux_conditions,
            ref_conditions=ref_conditions,
            router_conditions=router_conditions,
            redux_scale=redux_scale,
            redux_start_at=redux_start_at,
            redux_end_at=redux_end_at,
            ref_shift=ref_shift,
            ref_mask=ref_mask,
            ref_start_at=ref_start_at,
            ref_end_at=ref_end_at,
            enable_ref_cache=enable_ref_cache,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            generator=generator,
            **params
        ).images

        for attn_processor in self.flux_pipeline.transformer.attn_processors.values():
            if isinstance(attn_processor, AnyStoryFluxAttnProcessor2_0):
                for values in attn_processor.ref_bank.values():
                    del values
                attn_processor.ref_bank = {}
        torch.cuda.empty_cache()

        return images[0]


def prepare_params(
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


@torch.no_grad()
def generate(
        pipeline: FluxPipeline,
        redux_conditions: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        ref_conditions: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        router_conditions: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        redux_scale: float = 1.0,
        redux_start_at: float = 0.,
        redux_end_at: float = 1.,
        ref_shift: float = 0.0,
        ref_mask: torch.Tensor = None,
        ref_start_at: float = 0.,
        ref_end_at: float = 1.,
        enable_ref_cache: bool = False,
        **params,
):
    self = pipeline
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 4.1. Prepare conditions
    redux_embeds, redux_ids = ([] for _ in range(2))
    use_redux_cond = redux_conditions is not None and len(redux_conditions) > 0
    if use_redux_cond:
        for redux_condition in redux_conditions:
            tokens, ids = redux_condition
            redux_embeds.append(tokens * redux_scale)  # [bs, 81, 4096]
            redux_ids.append(ids)  # [81, 3]
        redux_embeds = torch.cat(redux_embeds, dim=1)
        redux_ids = torch.cat(redux_ids, dim=0)

    ref_latents, ref_ids = ([] for _ in range(2))
    use_ref_cond = ref_conditions is not None and len(ref_conditions) > 0
    if use_ref_cond:
        for ref_condition in ref_conditions:
            tokens, ids = ref_condition
            ref_latents.append(tokens)  # [bs, 1024, 4096]
            ref_ids.append(ids)  # [1024, 3]
        ref_latents = torch.cat(ref_latents, dim=1)
        ref_ids = torch.cat(ref_ids, dim=0)

    router_embeds, router_ids = ([] for _ in range(2))
    use_router = router_conditions is not None and len(router_conditions) > 0
    if use_router:
        for router_condition in router_conditions:
            tokens, ids = router_condition
            router_embeds.append(tokens)  # [bs, 81, 4096]
            router_ids.append(ids)  # [81, 3]
        router_embeds = torch.cat(router_embeds, dim=1)
        router_ids = torch.cat(router_ids, dim=0)

    num_conds = 0
    if use_redux_cond or use_ref_cond:
        num_conds = len(redux_conditions) or len(ref_conditions)

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            # handle conditions
            redux_start_step = int(num_inference_steps * redux_start_at + 0.5)
            redux_end_step = int(num_inference_steps * redux_end_at + 0.5)
            ref_start_step = int(num_inference_steps * ref_start_at + 0.5)
            ref_end_step = int(num_inference_steps * ref_end_at + 0.5)

            act_redux_cond = use_redux_cond and redux_start_step <= i < redux_end_step
            act_ref_cond = use_ref_cond and ref_start_step <= i < ref_end_step

            model_config = {}
            model_config["cache_ref"] = use_ref_cond and enable_ref_cache and i == ref_start_step
            model_config["use_ref_cache"] = use_ref_cond and enable_ref_cache and (
                    ref_start_step < i < ref_end_step
            )
            model_config["ref_shift"] = ref_shift
            model_config["ref_mask"] = ref_mask
            model_config["num_conds"] = num_conds if act_redux_cond or act_ref_cond else 0

            noise_pred = tranformer_forward(
                self.transformer,
                redux_hidden_states=redux_embeds if act_redux_cond else None,
                redux_ids=redux_ids if act_redux_cond else None,
                ref_hidden_states=ref_latents if act_ref_cond else None,
                ref_ids=ref_ids if act_ref_cond else None,
                router_hidden_states=router_embeds if use_router and (act_redux_cond or act_ref_cond) else None,
                router_ids=router_ids if use_router and (act_redux_cond or act_ref_cond) else None,
                model_config=model_config, hidden_states=latents, timestep=timestep / 1000,
                guidance=guidance, pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs, return_dict=False
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
                          latents / self.vae.config.scaling_factor
                  ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
