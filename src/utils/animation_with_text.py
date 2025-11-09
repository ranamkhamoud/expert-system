from PIL import Image
import numpy as np
import os
import soundfile as sf
import subprocess
from moviepy.editor import VideoClip

# Get ffmpeg path and set for moviepy
ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
print("IMAGEIO_FFMPEG_EXE set to:", os.environ["IMAGEIO_FFMPEG_EXE"])


def _golden_slider_blend(frame, slider_x, glow_radius=14, transparency=0.6, t=0.0):
    """
    Apply a warm transparent yellow glowing vertical bar at x = slider_x
    over the entire frame height.
    """
    h, w, _ = frame.shape
    frame = frame.astype(np.float32)

    core_color = np.array([255, 220, 100], dtype=np.float32)  # warm yellow core
    glow_color = np.array([255, 200, 80], dtype=np.float32)   # softer outer glow

    # Breathing pulse
    pulse = 0.3 + 0.3 * np.sin(2 * np.pi * 0.5 * t)

    for offset in range(-glow_radius, glow_radius + 1):
        alpha = np.exp(-abs(offset) / (glow_radius / 3.0)) * pulse * transparency
        color = glow_color if abs(offset) > 1 else core_color
        x = slider_x + offset
        if 0 <= x < w:
            frame[:, x] = (1 - alpha) * frame[:, x] + alpha * color

    return np.clip(frame, 0, 255).astype(np.uint8)


def create_animation_with_text(image_path1, image_path2, audio_path, output_path, image_prompt, audio_prompt):
    """
    Two-image version (original signature kept).
    Now: stacks two images vertically, no text, no white frame,
    with a warm transparent golden glow slider moving across.
    """
    # Hyperparameters
    space_between_images = 30

    # Load images
    image1 = np.array(Image.open(image_path1).convert('RGB'))
    image2 = np.array(Image.open(image_path2).convert('RGB'))
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape

    max_width = max(w1, w2)
    total_height = h1 + space_between_images + h2

    # Load audio
    audio_data, sample_rate = sf.read(audio_path)
    video_duration = len(audio_data) / sample_rate

    def make_frame(t):
        # Slider x position
        slider_x = int(t * (max_width - 1) / video_duration)
        slider_x = np.clip(slider_x, 0, max_width - 1)

        # Start with black frame then place both images
        frame = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        # First image at top
        frame[0:h1, 0:w1] = image1

        # Second image below with gap
        start2 = h1 + space_between_images
        frame[start2:start2 + h2, 0:w2] = image2

        # Apply golden glow slider across full height
        frame = _golden_slider_blend(frame, slider_x, glow_radius=14, transparency=0.6, t=t)
        return frame

    video_clip = VideoClip(make_frame, duration=video_duration)
    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None)

    # Copy original audio in
    os.system(f'ffmpeg -v quiet -y -i "{temp_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"')
    os.remove(temp_path)


def create_single_image_animation_with_text(image_path, audio_path, output_path, image_prompt, audio_prompt):
    """
    Single-image version (original signature kept).
    Shows only the image + warm transparent golden glow slider.
    No text, no white frame.
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = image.shape

    # Load audio
    audio_data, sample_rate = sf.read(audio_path)
    video_duration = len(audio_data) / sample_rate

    def make_frame(t):
        # Slider x position
        slider_x = int(t * (w - 1) / video_duration)
        slider_x = np.clip(slider_x, 0, w - 1)

        frame = image.copy()
        frame = _golden_slider_blend(frame, slider_x, glow_radius=14, transparency=0.6, t=t)
        return frame

    video_clip = VideoClip(make_frame, duration=video_duration)
    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None)

    # Copy original audio in
    os.system(f'ffmpeg -v quiet -y -i "{temp_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"')
    os.remove(temp_path)
