import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import time

# Define the text prompt for video generation
prompt = "Spider-Man battles a giant robotic villain in Times Square, dodging laser blasts. He web-slings around the enemy and delivers a powerful kick. Ultra high resolution 4k."


negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

# Track total execution time
total_start_time = time.time()

# Measure model loading time
model_load_start = time.time()
pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)  # Load model in FP16
pipe.to("cuda")  
model_load_end = time.time()
print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds.")

# Optimize model memory usage
pipe.enable_model_cpu_offload()  # Moves parts of the model to CPU when not in use
pipe.enable_vae_tiling()  # Reduces memory use by processing VAE in tiles

# Measure inference time
inference_start = time.time()
with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):  # Enable mixed precision inference
    video = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                 guidance_scale=7.5,
                 num_videos_per_prompt=1,
                 num_inference_steps=50,
                 num_frames=82).frames[0]
inference_end = time.time()
print(f"Inference completed in {inference_end - inference_start:.2f} seconds.")

# Measure video export time
export_start = time.time()
video_name = "genmo101.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path, fps=16)
export_end = time.time()
print(f"Video exported in {export_end - export_start:.2f} seconds.")

# Track total execution time
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

print(f"Video saved at {video_path}")
