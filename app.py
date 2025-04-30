import gradio as gr
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

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

# load matting model
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()


def image_matting(pil_image, pil_mask):
    def extract_mask(pil_image):
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(pil_image).unsqueeze(0).to('cuda')

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(pil_image.size)
        return mask

    if pil_mask is not None:
        assert pil_image.size == pil_mask.size, f"pil_image.size={pil_image.size}, pil_mask.size={pil_mask.size}"
        mask = ((np.array(pil_mask) > 200) * 255).astype(np.uint8)
    else:
        if pil_image.mode == "RGBA":
            mask = np.array(pil_image)[..., -1] > 200
            if np.all(mask):
                mask = ((np.array(extract_mask(pil_image.convert("RGB"))) > 200) * 255).astype(np.uint8)
            else:
                mask = ((np.array(pil_image)[..., -1] > 200) * 255).astype(np.uint8)
        else:
            mask = ((np.array(extract_mask(pil_image.convert("RGB"))) > 200) * 255).astype(np.uint8)
    non_zero_indices = np.nonzero(mask)
    min_x = np.min(non_zero_indices[1])
    max_x = np.max(non_zero_indices[1])
    min_y = np.min(non_zero_indices[0])
    max_y = np.max(non_zero_indices[0])

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    s = max(max_x - min_x, max_y - min_y)
    s = s * 1.04
    min_x = cx - s // 2
    min_y = cy - s // 2
    max_x = min_x + s
    max_y = min_y + s

    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.crop((min_x, min_y, max_x, max_y))
    pil_mask = Image.fromarray(mask).crop((min_x, min_y, max_x, max_y))

    np_image = np.array(pil_image)
    np_mask = np.array(pil_mask)[..., None] / 255.
    pil_masked_image = Image.fromarray((np_mask * np_image + (1 - np_mask) * 255.).astype(np.uint8))
    return pil_masked_image, pil_mask


def process(
        pil_subject_A_image=None,
        pil_subject_A_mask=None,
        pil_subject_B_image=None,
        pil_subject_B_mask=None,
        prompt="",
        inference_steps=25,
        guidance_scale=3.5,
        seed=-1,
):
    subject_images = []
    subject_masks = []
    if pil_subject_A_image is not None:
        pil_subject_A_image, pil_subject_A_mask = image_matting(pil_subject_A_image, pil_subject_A_mask)
        subject_images.append(pil_subject_A_image)
        subject_masks.append(pil_subject_A_mask)
    if pil_subject_B_image is not None:
        pil_subject_B_image, pil_subject_B_mask = image_matting(pil_subject_B_image, pil_subject_B_mask)
        subject_images.append(pil_subject_B_image)
        subject_masks.append(pil_subject_B_mask)

    if len(subject_images) > 1:
        # multi cond
        # enable_router: alleviate the blending issue of multiple subjects with similar semantics.
        enable_router = True
        ref_start_at = 0.09
    else:
        enable_router = False  # unnecessary; save computation
        ref_start_at = 0.0
    image = story_pipe.generate(prompt=prompt, images=subject_images, masks=subject_masks, seed=seed,
                                enable_router=enable_router, ref_start_at=ref_start_at,
                                num_inference_steps=inference_steps, height=512, width=512,
                                guidance_scale=guidance_scale)

    return image


def interface():
    with gr.Row(variant="panel"):
        gr.HTML(description + "<br>" + tips)
    with gr.Row(variant="panel"):
        with gr.Column(scale=2, min_width=100):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=100):
                    with gr.Tab(label="Subject [A]"):
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=100):
                                    pil_subject_A_image = gr.Image(type="pil", label="Subject [A] Reference Image",
                                                                   format="png", show_label=True, image_mode="RGBA")
                                with gr.Column(min_width=100):
                                    pil_subject_A_mask = gr.Image(type="pil",
                                                                  label="Subject [A] Mask (optional)",
                                                                  format="png", show_label=True, image_mode="L")

                with gr.Column(scale=1, min_width=100):
                    with gr.Tab(label="Subject [B]"):
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                with gr.Column(min_width=100):
                                    pil_subject_B_image = gr.Image(type="pil", label="Subject [B] Reference Image",
                                                                   format="png", show_label=True, image_mode="RGBA")
                                with gr.Column(min_width=100):
                                    pil_subject_B_mask = gr.Image(type="pil",
                                                                  label="Subject [B] Mask (optional)",
                                                                  format="png", show_label=True, image_mode="L")

            with gr.Group():
                prompt = gr.Textbox(value="", label='Prompt', lines=5, show_label=True)

            with gr.Accordion(label="Advanced Options", open=False):
                seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=2147483647, step=1,
                                 value=-1)
                inference_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, visible=True)
                guidance_scale = gr.Slider(label="guidance_scale", minimum=1.0, maximum=12.0,
                                           step=0.01, value=3.5)

        with gr.Column(scale=1, min_width=100):
            result_gallery = gr.Image(format="png", type="pil", label="Generated Image", visible=True, height=450)
            run_button = gr.Button(value="🧑‍🎨 RUN")

    run_button.click(
        fn=process,
        inputs=[pil_subject_A_image, pil_subject_A_mask, pil_subject_B_image, pil_subject_B_mask, prompt,
                inference_steps, guidance_scale, seed],
        outputs=[result_gallery]
    )

    with gr.Row():
        examples = [
            [
                "assets/examples/1.webp",
                "assets/examples/1_mask.webp",
                None,
                None,
                "Cartoon style. A sheep is riding a skateboard and gliding through the city, holding a wooden sign that says \"hello\".",
                "assets/examples/1_output.webp",
            ],
            [
                "assets/examples/2.webp",
                "assets/examples/2_mask.webp",
                None,
                None,
                "Cartoon style. Sun Wukong stands on a tank, holding up an ancient wooden sign high in the air. The sign reads 'AnyStory'. The background is a cyberpunk-style city sky filled with towering buildings.",
                "assets/examples/2_output.webp",
            ],
            [
                "assets/examples/3.webp",
                "assets/examples/3_mask.webp",
                None,
                None,
                "A modern and stylish Nezha playing an electric guitar, dynamic pose, vibrant colors, fantasy atmosphere, mythical Chinese character with a rock-and-roll twist, red scarf flowing in the wind, traditional elements mixed with contemporary design, cinematic lighting, 4k resolution",
                "assets/examples/3_output.webp",
            ],
            [
                "assets/examples/4.webp",
                "assets/examples/4_mask.webp",
                None,
                None,
                "a man riding a bike on the road",
                "assets/examples/4_output.webp",
            ],
            [
                "assets/examples/7.webp",
                "assets/examples/7_mask.webp",
                None,
                None,
                "Nezha is surrounded by a mysterious purple glow, with a pair of eyes glowing with an eerie red light. Broken talismans and debris float around him, highlighting his demonic nature and authority.",
                "assets/examples/7_output.webp",
            ],
            [
                "assets/examples/8.webp",
                "assets/examples/8_mask.webp",
                None,
                None,
                "The car is driving through a cyberpunk city at night in the middle of a heavy downpour.",
                "assets/examples/8_output.webp",
            ],
            [
                "assets/examples/9.webp",
                "assets/examples/9_mask.webp",
                None,
                None,
                "This cosmetic is placed on a table covered with roses.",
                "assets/examples/9_output.webp",
            ],
            [
                "assets/examples/10.webp",
                "assets/examples/10_mask.webp",
                None,
                None,
                "A little boy model is posing for a photo.",
                "assets/examples/10_output.webp",
            ],
            [
                "assets/examples/6_1.webp",
                "assets/examples/6_1_mask.webp",
                "assets/examples/6_2.webp",
                "assets/examples/6_2_mask.webp",
                "Two men are sitting by a wooden table, which is laden with delicious food and a pot of wine. One of the men holds a wine glass, drinking heartily with a bold expression; the other smiles as he pours wine for his companion, both of them engaged in cheerful conversation. In the background is an ancient pavilion surrounded by emerald bamboo groves, with sunlight filtering through the leaves to cast dappled shadows.",
                "assets/examples/6_output.webp",
            ],
        ]
        gr.Examples(
            label="Examples",
            examples=examples,
            inputs=[pil_subject_A_image, pil_subject_A_mask, pil_subject_B_image, pil_subject_B_mask, prompt,
                    result_gallery],
        )


if __name__ == "__main__":
    title = r"""
            <div style="text-align: center;">
                <h1> AnyStory: Towards Unified Single and Multiple Subject Personalization in Text-to-Image Generation </h1>
                <h1> V2.0.0 </h1>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href="https://arxiv.org/pdf/2501.09503"><img src="https://img.shields.io/badge/arXiv-2501.09503-red"></a>
                    &nbsp;
                    <a href='https://aigcdesigngroup.github.io/AnyStory/'><img src='https://img.shields.io/badge/Project_Page-AnyStory-green' alt='Project Page'></a>
                    &nbsp;
                    <a href='https://modelscope.cn/studios/iic/AnyStory'><img src='https://img.shields.io/badge/Demo-ModelScope-blue'></a>
                </div>
                </br>
            </div>
        """

    description = r"""🚀🚀🚀 Quick Start:<br>
        1. Upload subject images (clean background; real human IDs unsupported for now), Add prompts (only EN supported), and Click "<b>RUN</b>".<br>
        2. (Optional) Upload B&W masks for subjects. This helps the model better reference the subject you specify.<br>
        """

    tips = r"""💡💡💡 Tips:<br>
    If the subject doesn't appear, try adding a detailed description of the subject in the prompt that matches the reference image, and avoid conflicting details (e.g., significantly altering the subject's appearance).<br>
    """

    block = gr.Blocks(title="AnyStory", theme=gr.themes.Ocean()).queue()
    with block:
        gr.HTML(title)

        interface()

    block.launch(server_name='0.0.0.0', share=False, server_port=9999, max_threads=10)
