import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
#from IPython.display import Video
#from IPython.display import HTML
import cv2
import numpy as np
import time

# Track total execution time
total_start_time = time.time()


# Track model loading time
start_model_load = time.time()

pipe = DiffusionPipeline.from_pretrained("VideoCrafter/VideoCrafter2", torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#for using GPU acceleration and CPU offloading
pipe = pipe.to("cuda")

end_model_load = time.time()
print(f"Model loaded in {end_model_load - start_model_load:.2f} seconds.")

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.vae.enable_tiling()

prompt = """"An old man sits alone on a weathered park bench, tossing crumbs to the pigeons as autumn leaves dance around him. He pulls his coat tighter against the chill, his eyes following the birds, yet his mind drifts to a time long gone. Generate output in ultra high resolution 4k."""

negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."""

# Start inference time tracking
start_inference_time = time.time()

# Generate video
video = pipe(prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        #max_sequence_length=512,
        num_inference_steps=50, 
        num_frames=122,
        generator = torch.Generator(device="cuda").manual_seed(42)).frames[0]

end_inference_time = time.time()
print(f"Inference completed in {end_inference_time - start_inference_time:.2f} seconds.")


#model saving
video_name="video_crafter101.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path,fps=24)


# Track total execution time
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

print(f"Video saved at {video_path}")                  

