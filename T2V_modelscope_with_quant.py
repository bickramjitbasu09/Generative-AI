import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os
import time


# Track total execution time
total_start_time = time.time()


# Track model loading time
start_model_load = time.time()

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.bfloat16, variant="bf16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

end_model_load = time.time()
print(f"Model loaded in {end_model_load - start_model_load:.2f} seconds.")

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()


# Define prompt
prompt = """Spider-Man battles a giant robotic villain in Times Square, dodging laser blasts. He web-slings around the enemy and delivers a powerful kick. Ultra high resolution 4k."""


# Start inference time tracking
start_inference_time = time.time()

# Generate video
video = pipe(prompt=prompt, num_inference_steps=50, num_frames=82).frames[0]

end_inference_time = time.time()
print(f"Inference completed in {end_inference_time - start_inference_time:.2f} seconds.")


#model saving
video_name = "modelscope101.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path, fps=16)

# Track total execution time
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

print(f"Video saved at {video_path}")