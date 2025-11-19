import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, AllegroTransformer3DModel, AllegroPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
import time

# Track total execution time
total_start_time = time.time()


# Track model loading time
start_model_load = time.time()


pipe = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    torch_dtype=torch.bfloat16,
   # device_map="balanced",
)
pipe.to("cuda")

end_model_load = time.time()
print(f"Model loaded in {end_model_load - start_model_load:.2f} seconds.")

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.vae.enable_tiling()


prompt ="A young woman standing in the rain, staring at a letter in her hand as tears mix with the raindrops on her face.Realistic output in Ultra high resolution 4k."


negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""
          
                  
# Start inference time tracking
start_inference_time = time.time()

# Generate video
video = pipe(prompt=prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        max_sequence_length=512,
        num_inference_steps=50, 
        num_frames=42,
        generator = torch.Generator(device="cuda").manual_seed(42)).frames[0]

end_inference_time = time.time()
print(f"Inference completed in {end_inference_time - start_inference_time:.2f} seconds.")


#model saving
video_name="allegro101.mp4"
video_path = f"./{video_name}"
export_to_video(video, output_video_path=video_path,fps=8)


# Track total execution time
total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

print(f"Video saved at {video_path}")                  

