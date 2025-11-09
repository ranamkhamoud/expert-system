from PIL import Image
import numpy as np
import os
import soundfile as sf
import subprocess
from omegaconf import OmegaConf
from moviepy.editor import VideoClip

# Setup ffmpeg path
ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path


def create_single_image_animation_with_text(image_path, audio_path, output_path, image_prompt, audio_prompt):
    """
    Single image + transparent yellow glow slider (soft and cinematic).
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    image_height, image_width, _ = image.shape

    # Load audio
    audio_data, sample_rate = sf.read(audio_path)
    video_duration = len(audio_data) / sample_rate

    # Slider parameters
    slider_width = 4
    glow_radius = 14          # softness of glow
    core_color = np.array([255, 220, 100])  # warm yellow core
    glow_color = np.array([255, 200, 80])   # slightly softer outer glow

    def make_frame(t):
        # Slider position
        slider_x = int(t * (image_width - slider_width // 2) / video_duration)
        slider_x = np.clip(slider_x, 0, image_width - slider_width)

        # Copy image to float for blending
        frame = image.copy().astype(np.float32)

        # Pulse for breathing glow
        pulse = 0.3 + 0.3 * np.sin(2 * np.pi * 0.5 * t)

        # Apply glowing bar with transparency
        for offset in range(-glow_radius, glow_radius + 1):
            alpha = np.exp(-abs(offset) / (glow_radius / 3.0)) * pulse * 0.6  # reduce opacity
            color = glow_color if abs(offset) > 1 else core_color
            x = slider_x + offset
            if 0 <= x < image_width:
                frame[:, x] = (1 - alpha) * frame[:, x] + alpha * color

        # Gentle overall transparency blending
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    # Create video
    video_clip = VideoClip(make_frame, duration=video_duration)
    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None)

    # Merge original audio
    os.system(f"ffmpeg -v quiet -y -i \"{temp_path}\" -i \"{audio_path}\" -c:v copy -c:a aac \"{output_path}\"")
    os.remove(temp_path)


# Example usage
if __name__ == '__main__':
    example_path = '/home/czyang/Workspace/images-that-sound/logs/soundify-denoise/colorization/tiger_example_06'
    spec = f'{example_path}/spec.png'
    audio = f'{example_path}/audio.wav'
    config_path = f'{example_path}/config.yaml'
    cfg = OmegaConf.load(config_path)
    image_prompt = cfg.trainer.image_prompt
    audio_prompt = cfg.trainer.audio_prompt
    output_path = 'test_goldglow.mp4'

    create_single_image_animation_with_text(spec, audio, output_path, image_prompt, audio_prompt)
