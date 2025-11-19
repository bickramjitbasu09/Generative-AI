
import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import numpy as np
from PIL import Image

# Enable TF32 for better BF16 performance (optional)
torch.backends.cuda.matmul.allow_tf32 = True

# Start tracking total execution time
total_start_time = time.time()

prompt = """ A man walks through a bustling city street at night.Realistic output in Ultra high resolution 4k."""

### Load First Diffusion Model ###
start_time = time.time()
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.bfloat16)
pipe.to("cuda")  # Move model to GPU
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
first_model_load_time = time.time() - start_time
print(f"First model loaded in {first_model_load_time:.2f} seconds.")

# Generate video frames
inference_start_time = time.time()
video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=122).frames
video_frames = [frame for frame in video_frames[0]]  # Split frames if necessary
inference_time_first = time.time() - inference_start_time
print(f"First model inference completed in {inference_time_first:.2f} seconds.")

# Ensure frames are uint8 type before using Image
video = [(np.array(frame) * 255).astype(np.uint8) for frame in video_frames]

# Resize frames using PIL
video = [Image.fromarray(frame).resize((1024, 576)) for frame in video]

### Load Second Diffusion Model ###
start_time = time.time()
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.bfloat16)
pipe.to("cuda")  # Move model to GPU
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
second_model_load_time = time.time() - start_time
print(f"Second model loaded in {second_model_load_time:.2f} seconds.")

# Generate video using the second model
inference_start_time = time.time()
video_frames2 = pipe(prompt, video=video, strength=0.6).frames
video_frames2 = [frame for frame in video_frames2[0]]
inference_time_second = time.time() - inference_start_time
print(f"Second model inference completed in {inference_time_second:.2f} seconds.")

# Export video
export_start_time = time.time()
video_name = "cerspense201.mp4"
video_path = f"./{video_name}"
export_to_video(video_frames2, output_video_path=video_path, fps=24)
export_time = time.time() - export_start_time
print(f"Video saved at {video_path}")
print(f"Video export completed in {export_time:.2f} seconds.")

# Print total execution time
total_end_time = time.time()
total_execution_time = total_end_time - total_start_time
print(f"Total execution time: {total_execution_time:.2f} seconds.")


