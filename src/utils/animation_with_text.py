from PIL import Image
import numpy as np
import os
import soundfile as sf
import subprocess
from moviepy.editor import VideoClip


# ---------------------------------------------------------
# Set ffmpeg path for moviepy
# ---------------------------------------------------------
def _setup_ffmpeg():
    try:
        ffmpeg_path = subprocess.check_output(['which', 'ffmpeg']).decode().strip()
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        print("IMAGEIO_FFMPEG_EXE set to:", os.environ["IMAGEIO_FFMPEG_EXE"])
    except Exception as e:
        print("Warning: could not automatically find ffmpeg:", e)


_setup_ffmpeg()


# ---------------------------------------------------------
# Golden slider overlay
# ---------------------------------------------------------
def _golden_slider_blend(frame, slider_x, glow_radius=14, transparency=0.6, t=0.0):
    """
    Draw a warm golden vertical glow bar centered at slider_x
    over the full height of the frame.
    """
    h, w, _ = frame.shape
    frame = frame.astype(np.float32)

    core_color = np.array([255, 220, 100], dtype=np.float32)  # warm yellow core
    glow_color = np.array([255, 200, 80], dtype=np.float32)   # softer outer glow

    # Breathing pulse over time
    pulse = 0.3 + 0.3 * np.sin(2 * np.pi * 0.5 * t)

    for offset in range(-glow_radius, glow_radius + 1):
        alpha = np.exp(-abs(offset) / (glow_radius / 3.0)) * pulse * transparency
        color = glow_color if abs(offset) > 1 else core_color
        x = slider_x + offset
        if 0 <= x < w:
            frame[:, x] = (1 - alpha) * frame[:, x] + alpha * color

    return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------
# Helper: choose which image is the spectrogram
# ---------------------------------------------------------
def _select_spectrogram_image(image_path1, image_path2=None):
    """
    Prefer any path whose basename contains 'spec' (e.g., 'spec.png', 'myspec_0.png').
    If none match, fall back to image_path1.
    """
    candidates = [p for p in [image_path1, image_path2] if p is not None]

    # Try to find something like spec.png / spec_foo.png etc.
    for p in candidates:
        base = os.path.basename(p).lower()
        if "spec" in base:
            print(f"[animation] Using spectrogram image: {p}")
            return p

    # Fallback
    print(f"[animation] No 'spec' in filenames, using image_path1: {image_path1}")
    return image_path1


# ---------------------------------------------------------
# Core single-image video generator (ONLY the image)
# ---------------------------------------------------------
def _make_single_image_video(image_path, audio_path, output_path):
    """
    Core routine:
    - Loads a single image (your spectrogram)
    - Plays a golden slider from left to right over its width
    - Duration is exactly the audio length
    - Final video = ONLY the image + slider + audio
    """
    # Load image (spectrogram)
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = image.shape

    # Load audio to get duration
    audio_data, sample_rate = sf.read(audio_path)
    video_duration = len(audio_data) / sample_rate

    def make_frame(t):
        # Slider x position over time
        slider_x = int(t * (w - 1) / max(video_duration, 1e-6))
        slider_x = np.clip(slider_x, 0, w - 1)

        frame = image.copy()
        frame = _golden_slider_blend(frame, slider_x, glow_radius=14, transparency=0.6, t=t)
        return frame

    temp_path = output_path[:-4] + '-temp.mp4'
    video_clip = VideoClip(make_frame, duration=video_duration)
    video_clip.write_videofile(temp_path, codec='libx264', fps=60, logger=None)

    # Mux original audio into the video
    os.system(
        f'ffmpeg -v quiet -y -i "{temp_path}" -i "{audio_path}" '
        f'-c:v copy -c:a aac "{output_path}"'
    )
    os.remove(temp_path)


# ---------------------------------------------------------
# Public APIs (signatures kept exactly as original)
# ---------------------------------------------------------
def create_animation_with_text(image_path1, image_path2, audio_path, output_path, image_prompt, audio_prompt):
    """
    Two-image version (original signature kept for compatibility).

    NEW behavior:
    - Chooses whichever path (image_path1 or image_path2) looks like a spectrogram,
      preferring filenames that contain 'spec' (e.g., 'spec.png').
    - Ignores image_prompt and audio_prompt.
    - Output: that chosen image + golden slider + audio.
    """
    spectrogram_path = _select_spectrogram_image(image_path1, image_path2)
    _make_single_image_video(spectrogram_path, audio_path, output_path)


def create_single_image_animation_with_text(image_path, audio_path, output_path, image_prompt, audio_prompt):
    """
    Single-image version (original signature kept).

    Behavior:
    - Uses ONLY image_path (expected to be the spectrogram, e.g., 'spec.png').
    - Output: spectrogram + golden slider + audio.
    """
    print(f"[animation] Using single-image spectrogram: {image_path}")
    _make_single_image_video(image_path, audio_path, output_path)


# Optional: quick CLI test if you ever want to run this file directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectrogram + golden slider video generator")
    parser.add_argument("--image1", required=True, help="Path to first image (e.g. spec.png or generated image)")
    parser.add_argument("--image2", required=False, help="Optional second image (if present, will prefer the one with 'spec' in name)")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out", required=True, help="Output video path (e.g. output.mp4)")
    args = parser.parse_args()

    spec_img = _select_spectrogram_image(args.image1, args.image2)
    _make_single_image_video(spec_img, args.audio, args.out)
    print("Done:", args.out)
