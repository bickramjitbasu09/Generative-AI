import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
import time

# Start tracking total execution time
total_start_time = time.time()

# Measure model loading time
model_load_start = time.time()
pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")
model_load_end = time.time()
print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds.")

# Define the prompt
prompt = "A man walks through a bustling city street at night.Realistic output in Ultra high resolution 4k."


# Measure inference time
inference_start = time.time()
video = pipe(
    prompt=prompt,
    width=704,
    height=480,
    num_frames=122,
    num_inference_steps=50,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
).frames[0]
inference_end = time.time()
print(f"Inference completed in {inference_end - inference_start:.2f} seconds.")

# Measure video export time
export_start = time.time()
video_name = "LTX201.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path, fps=24)
export_end = time.time()
print(f"Video saved at {video_path}")
print(f"Video export completed in {export_end - export_start:.2f} seconds.")

# End tracking total execution time
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

