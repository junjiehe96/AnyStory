import gc
import json

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from anystory.generate import AnyStoryFluxPipeline


def parse_script(script_dict):
    alias_to_character_prompt = {}
    for character in script_dict["角色表"]:
        alias = character["alias"].strip()
        character_prompt = character["simple_prompt"].rstrip(".").strip()
        assert alias not in alias_to_character_prompt, f"duplicate character in script: {alias}"
        alias_to_character_prompt[alias] = character_prompt

    index_to_storyboard = {}
    storyboard_indices = []
    for storyboard in script_dict["分镜头"]:
        storyboard_index = storyboard["序号"]
        storyboard_prompt = storyboard["prompt"].strip()
        assert storyboard_index not in index_to_storyboard, f"duplicate storyboard index in script: {storyboard_index}"
        index_to_storyboard[storyboard_index] = {"prompt": storyboard_prompt}
        if "act_roles" in storyboard:
            index_to_storyboard[storyboard_index]["act_roles"] = storyboard["act_roles"]
        else:
            index_to_storyboard[storyboard_index]["act_roles"] = []
        storyboard_indices.append(storyboard_index)

    return alias_to_character_prompt, index_to_storyboard, storyboard_indices


def replace_alias_with_character_prompt(prompt, alias_to_character_prompts):
    act_aliases = []
    for alias in alias_to_character_prompts:
        if alias in prompt:
            prompt = prompt.replace(alias, alias_to_character_prompts[alias] + " ")
            act_aliases.append(alias)
    return prompt, act_aliases


def apply_style(style_name, prompt):
    styles = {
        "(No style)": "{}",
        "Chinese painting": "A traditional Chinese painting illustrating {}. ink wash style, mainly black and white, with negative space design, smooth lines, and soft brushstrokes, painted on rice paper using a traditional brush.",
        "Japanese Anime": "anime artwork illustrating {}. created by japanese anime studio. highly emotional. best quality, high resolution, Anime Style, Manga Style, concept art, webtoon",
        "Pixar/Disney Character": "Create a Disney Pixar 3D style illustration on {} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.",
        "Photographic": "cinematic photo {} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus",
        "Comic book": "comic {} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "Line art": "line art drawing {} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "Black and White Film Noir": "{} . b&w, Monochromatic, Film Photography, film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k",
        "Isometric Rooms": "Tiny cute isometric {} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render",
    }
    p = styles.get(style_name)
    return p.format(prompt)


class StoryboardPipeline:
    def __init__(self, device="cuda"):
        anystory_path = hf_hub_download(repo_id="Junjie96/AnyStory", filename="anystory_flux.bin")
        story_pipe = AnyStoryFluxPipeline(
            hf_flux_pipeline_path="black-forest-labs/FLUX.1-dev",
            hf_flux_redux_path="black-forest-labs/FLUX.1-Redux-dev",
            anystory_path=anystory_path,
            device=device,
            torch_dtype=torch.bfloat16
        )

        # load matting model
        birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
        birefnet.to(device)
        birefnet.eval()

        self.story_pipe = story_pipe
        self.birefnet = birefnet

        self.ind_to_storyboard_result = {}
        self.alias_to_character_image = {}

        self.device = device

    def image_matting(self, pil_image, pil_mask=None):
        def extract_mask(pil_image):
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_images = transform_image(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.birefnet(input_images)[-1].sigmoid().cpu()
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

        pil_image = pil_image.convert("RGB").crop((min_x, min_y, max_x, max_y))
        pil_mask = Image.fromarray(mask).crop((min_x, min_y, max_x, max_y))

        np_image = np.array(pil_image)
        np_mask = np.array(pil_mask)[..., None] / 255.
        pil_masked_image = Image.fromarray((np_mask * np_image + (1 - np_mask) * 255.).astype(np.uint8))
        return pil_masked_image, pil_mask

    def generate(self, pil_images, prompt, seed=-1):

        images = []
        masks = []
        for pil_image in pil_images:
            pil_masked_image, pil_mask = self.image_matting(pil_image)
            images.append(pil_masked_image)
            masks.append(pil_mask)

        if len(images) > 1:
            # multi cond
            # enable_router: alleviate the blending issue of multiple subjects with similar semantics.
            enable_router = True
            ref_start_at = 0.09
        else:
            enable_router = False  # unnecessary; save computation
            ref_start_at = 0.0
        result = self.story_pipe.generate(
            prompt=prompt,
            images=images,
            masks=masks,
            seed=seed,
            enable_router=enable_router, ref_start_at=ref_start_at,
            num_inference_steps=25, height=512, width=512,
            guidance_scale=3.5
        )
        return result

    def new_story(self):
        # a new story
        gc.collect()
        torch.cuda.empty_cache()
        self.ind_to_storyboard_result.clear()
        self.alias_to_character_image.clear()

    def __call__(self, script_dict, frame_index: list = None, alias_to_character_image: dict = None,
                 style_name: str = "Japanese Anime", seed: int = -1):

        alias_to_character_prompts, index_to_storyboard, storyboard_indices = parse_script(script_dict)

        if alias_to_character_image is not None:
            for alias in alias_to_character_image:
                if alias in self.alias_to_character_image:
                    print(f"{alias} character_image has been cached, skipping process.")
                    continue
                self.alias_to_character_image[alias] = alias_to_character_image[alias]

        for alias in alias_to_character_prompts:
            if alias not in self.alias_to_character_image:
                # create character by generation
                cur_prompt = alias_to_character_prompts[alias]
                print(alias, cur_prompt)
                cur_prompt = apply_style(style_name, cur_prompt)
                character_image = self.generate([], cur_prompt, seed=seed)
                self.alias_to_character_image[alias] = character_image

        if frame_index is None:
            gen_storyboard_indices = storyboard_indices
        else:
            gen_storyboard_indices = frame_index

        for storyboard_index in gen_storyboard_indices:
            storyboard_prompt = index_to_storyboard[storyboard_index]["prompt"]
            storyboard_prompt, act_aliases = \
                replace_alias_with_character_prompt(storyboard_prompt, alias_to_character_prompts)
            if "act_roles" in index_to_storyboard[storyboard_index]:
                act_aliases = list(set(act_aliases + index_to_storyboard[storyboard_index]["act_roles"]))

            print(act_aliases, storyboard_index, storyboard_prompt)

            storyboard_prompt = apply_style(style_name, storyboard_prompt)
            character_images = [self.alias_to_character_image[alias] for alias in act_aliases]
            result = self.generate(character_images, storyboard_prompt, seed=seed)
            self.ind_to_storyboard_result[storyboard_index] = result

        total_results = {}
        total_results.update(self.ind_to_storyboard_result)
        for alias in self.alias_to_character_image:
            total_results.update(
                {alias: self.alias_to_character_image[alias]}
            )

        return total_results


if __name__ == "__main__":
    storyboard_pipe = StoryboardPipeline()

    script_dict = json.load(open("assets/scripts/013420.json"))
    print(script_dict)
    results = storyboard_pipe(script_dict, style_name="Comic book")
    print(results)

    for key, result in results.items():
        result.save(f"output_{key}.png")
