from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
import pandas as pd
import os
import random
import numpy as np
import argparse

def get_negative_images(df):
    return list(df[df["label"] == "negative"]["path"])

def get_positive_images_maks(df):
    return [p.replace('.png', '_GT.png') for p in list(df[df["label"] == "positive"]["path"])]

def main(args):
    
    ### ARGUMENTS
    src_dir = args.src_dir
    imgs_per_prompt = args.imgs_per_prompt
    seed = args.seed

    # prompts used in the paper
    prompts = ["white marks on the wall", "copper metal scratches"]
    negative_prompt="smooth, plain, black, dark, shadow"

    dst_dir = os.path.join(src_dir, f"augmented_{imgs_per_prompt*len(prompts)}")
    os.makedirs(dst_dir, exist_ok=True)

    # hyperparameters used in the paper
    num_inference_steps = 30
    guidance_scale = 20.0
    strength = 0.99
    padding_mask_crop = 2
    RES = (224, 632)
    # needed to ovecome the sdxl shape bias
    TARGET = (1024, 1024)


    ### MAIN
    # cuda if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device: ", device)

    # seed everything
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # create directories
    os.makedirs(os.path.join(dst_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'masks'), exist_ok=True)

    df_path = os.path.join(src_dir, 'train.csv')
    df = pd.read_csv(df_path)
    negative_imgs = get_negative_images(df)
    positive_masks = get_positive_images_maks(df)
    print(f'Num negative images: {len(negative_imgs)}')
    print(f'Num positive masks: {len(positive_masks)}')

    model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    print(f'Loading model {model}')
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16").to(device)

    img_idx = 0
    for prompt in prompts:
        print(f'Generating images for prompt: {prompt}')
        cnt = 0
        for cnt in range(imgs_per_prompt):
            # by sampling 1 by 1 we can generate more anomalies than what we have in the dataset (246)
            mask_name = random.sample(positive_masks, 1)[0]
            mask_path = os.path.join(src_dir, 'train', mask_name)
            neg_img_name = random.sample(negative_imgs, 1)[0]
            neg_img_path = os.path.join(src_dir, 'train', neg_img_name)
            
            neg_img = load_image(neg_img_path).resize(TARGET)
            mask = load_image(mask_path).resize(TARGET)

            out_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=neg_img,
                mask_image=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,  # steps between 15 and 30 work well for us
                strength=strength,  # make sure to use `strength` below 1.0
                generator=generator,
                height=TARGET[1],
                width=TARGET[0],
                original_size = TARGET,
                target_size = TARGET,
                padding_mask_crop = padding_mask_crop
            ).images[0]
            out_image = out_image.resize(RES)
            mask = mask.resize(RES)
            # save the image with progressive name
            out_img_path = f'{dst_dir}/imgs/{str(img_idx + cnt).zfill(5)}.png'
            out_image.save(out_img_path)
            # save the mask with progressive name
            out_mask_path = f'{dst_dir}/masks/{str(img_idx + cnt).zfill(5)}.png'
            mask.save(out_mask_path)
            cnt += 1
        img_idx += cnt

if __name__ == "__main__":
    # ARGUMENTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing the preprocessed dataset")
    parser.add_argument("--imgs_per_prompt", type=int, default=50, help="Number of images to generate per prompt")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random generation")
    args = parser.parse_args()

    main(args)