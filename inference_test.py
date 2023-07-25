from diffusers import StableDiffusionPipeline
import torch
from time import time_ns

SEED = time_ns() % 2**32
model_path = "./outputs"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, requires_safety_checker=False)
pipe.to("cuda")


torch.manual_seed(SEED)
prompt = "A [V] girl face, photograph"
image = pipe(prompt, num_inference_steps=30, guidance_scale=5).images[0]
image.save("test_before.png")

pipe.unet.load_attn_procs(model_path)
torch.manual_seed(SEED)
image = pipe(prompt, num_inference_steps=30, guidance_scale=5).images[0]
image.save("test_after.png")