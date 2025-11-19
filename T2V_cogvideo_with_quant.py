import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import time

prompt = "A young woman standing in the rain, staring at a letter in her hand as tears mix with the raindrops on her face."

start_time = time.time()
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")  # Move model to GPU (ROCm maps "cuda" to AMD GPUs)
end_time = time.time()

# Print model loading time
print(f"Model loaded in {end_time - start_time:.2f} seconds.")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=40,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

#export_to_video(video, "output.mp4", fps=8)

video_name="cogvideo_output3.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path,fps=8)
