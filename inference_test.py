from diffusers import StableDiffusionPipeline
import torch

model_path = "./outputs"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A [V] face"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("test.png")