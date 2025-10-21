
"""
tasks/task_c.py

Task (c): Text-to-Image / Image-to-Image / Inpainting with Hugging Face Diffusers.

We use:
- Text-to-Image: stabilityai/sd-turbo (fast) or runwayml/stable-diffusion-v1-5
- Image-to-Image: StableDiffusionImg2ImgPipeline
- Inpainting: StableDiffusionInpaintPipeline

NOTE: GPU strongly recommended. CPU works but is slow.
"""
from __future__ import annotations

import argparse, os, torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import numpy as np
import random

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser(description="Task (c): Diffusers pipelines (text2img / img2img / inpaint).")
    ap.add_argument("--mode", choices=["text2img", "img2img", "inpaint"], required=True, help="Which pipeline to run.")
    ap.add_argument("--prompt", type=str, default="A futuristic Ukrainian cityscape at golden hour, ultra-detailed, 4k", help="Text prompt.")
    ap.add_argument("--negative-prompt", type=str, default="", help="Negative prompt (optional).")
    ap.add_argument("--steps", type=int, default=20, help="Inference steps.")
    ap.add_argument("--guidance", type=float, default=2.5, help="CFG scale (classifier-free guidance).")
    ap.add_argument("--width", type=int, default=512, help="Output width.")
    ap.add_argument("--height", type=int, default=512, help="Output height.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--model", type=str, default="stabilityai/sd-turbo", help="Base model name for text2img/img2img/inpaint.")
    ap.add_argument("--init-image", type=str, default=None, help="Path to init image for img2img.")
    ap.add_argument("--mask-image", type=str, default=None, help="Path to mask image (white=painted area) for inpaint.")
    ap.add_argument("--strength", type=float, default=0.6, help="How much to transform init image in img2img (0..1).")
    ap.add_argument("--outdir", type=str, default="outputs/task_c", help="Where to save images.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    if args.mode == "text2img":
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)
        img = pipe(args.prompt, negative_prompt=args.negative_prompt or None, num_inference_steps=args.steps, guidance_scale=args.guidance, width=args.width, height=args.height).images[0]
        out_path = os.path.join(args.outdir, "text2img.png")
        img.save(out_path)
        print("Saved:", out_path)
    elif args.mode == "img2img":
        assert args.init_image, "--init-image is required for img2img"
        init_img = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)
        img = pipe(prompt=args.prompt, image=init_img, strength=args.strength, guidance_scale=args.guidance, num_inference_steps=args.steps).images[0]
        out_path = os.path.join(args.outdir, "img2img.png")
        img.save(out_path)
        print("Saved:", out_path)
    else:  # inpaint
        assert args.init_image and args.mask_image, "--init-image and --mask-image are required for inpaint"
        init_img = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
        mask_img = Image.open(args.mask_image).convert("RGB").resize((args.width, args.height))
        pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        pipe = pipe.to(device)
        img = pipe(prompt=args.prompt, image=init_img, mask_image=mask_img, strength=args.strength, guidance_scale=args.guidance, num_inference_steps=args.steps).images[0]
        out_path = os.path.join(args.outdir, "inpaint.png")
        img.save(out_path)
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
