import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from anystory.generate import AnyStoryFluxPipeline

anystory_path = hf_hub_download(repo_id="Junjie96/AnyStory", filename="anystory_flux.bin")
story_pipe = AnyStoryFluxPipeline(
    hf_flux_pipeline_path="black-forest-labs/FLUX.1-dev",
    hf_flux_redux_path="black-forest-labs/FLUX.1-Redux-dev",
    anystory_path=anystory_path,
    device="cuda",
    torch_dtype=torch.bfloat16
)
# you can add lora here
# story_pipe.flux_pipeline.load_lora_weights(lora_path, adapter_name="...")

# single-subject
subject_image = Image.open("assets/examples/1.webp").convert("RGB")
subject_mask = Image.open("assets/examples/1_mask.webp").convert("L")
prompt = "Cartoon style. A sheep is riding a skateboard and gliding through the city," \
         " holding a wooden sign that says \"hello\"."
image = story_pipe.generate(prompt=prompt, images=[subject_image], masks=[subject_mask], seed=2025,
                            num_inference_steps=25, height=512, width=512,
                            guidance_scale=3.5)
image.save("output_1.png")

# multi-subject
subject_image_1 = Image.open("assets/examples/6_1.webp").convert("RGB")
subject_mask_1 = Image.open("assets/examples/6_1_mask.webp").convert("L")
subject_image_2 = Image.open("assets/examples/6_2.webp").convert("RGB")
subject_mask_2 = Image.open("assets/examples/6_2_mask.webp").convert("L")
prompt = "Two men are sitting by a wooden table, which is laden with delicious food and a pot of wine. " \
         "One of the men holds a wine glass, drinking heartily with a bold expression; " \
         "the other smiles as he pours wine for his companion, both of them engaged in cheerful conversation. " \
         "In the background is an ancient pavilion surrounded by emerald bamboo groves, with sunlight filtering " \
         "through the leaves to cast dappled shadows."

image = story_pipe.generate(prompt=prompt,
                            images=[subject_image_1, subject_image_2],
                            masks=[subject_mask_1, subject_mask_2],
                            seed=2025,
                            enable_router=True, ref_start_at=0.09,
                            num_inference_steps=25, height=512, width=512,
                            guidance_scale=3.5)
image.save("output_2.png")
