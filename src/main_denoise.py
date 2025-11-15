import os
from typing import Optional
from tqdm import tqdm
import warnings
import soundfile as sf
import numpy as np
import shutil
import glob
import copy
import matplotlib.pyplot as plt

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import torch
import torch.nn as nn
import librosa
from transformers import logging
from lightning import seed_everything

from torchvision.utils import save_image


import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.rich_utils import print_config_tree
from src.utils.animation_with_text import create_animation_with_text, create_single_image_animation_with_text
from src.utils.re_ranking import select_top_k_ranking, select_top_k_clip_ranking
from src.utils.pylogger import RankedLogger
from src.utils.consistency_check import wav2spec, inverse_stft
log = RankedLogger(__name__, rank_zero_only=True)


def save_audio(audio, save_path):
    sf.write(save_path, audio, samplerate=16000)


def encode_prompt(prompt, diffusion_guidance, device, negative_prompt='', time_repeat=1):
    '''Encode text prompts into embeddings 
    '''
    prompts = [prompt] * time_repeat
    negative_prompts = [negative_prompt] * time_repeat

    # Prompts -> text embeds
    cond_embeds = diffusion_guidance.get_text_embeds(prompts, device) # [B, 77, 768]
    uncond_embeds = diffusion_guidance.get_text_embeds(negative_prompts, device) # [B, 77, 768]
    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0) # [2 * B, 77, 768]
    return text_embeds

def estimate_noise(diffusion, latents, t, text_embeddings, guidance_scale): 
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    # predict the noise residual
    noise_pred = diffusion.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

    # perform guidance
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    return noise_pred


@hydra.main(version_base="1.3", config_path="../configs/main_denoise", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for training
    """

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
        logging.set_verbosity_error()
    
    if cfg.extras.get("print_config"):
        print_config_tree(cfg, resolve=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log.info(f"Instantiating Image Diffusion model <{cfg.image_diffusion_guidance._target_}>")
    image_diffusion_guidance = hydra.utils.instantiate(cfg.image_diffusion_guidance).to(device)

    # always instantiate audio diffusion guidance for decoding spectrograms from latents
    log.info(f"Instantiating Audio Diffusion guidance model <{cfg.audio_diffusion_guidance._target_}>")
    audio_diffusion_guidance = hydra.utils.instantiate(cfg.audio_diffusion_guidance).to(device)

    # create the shared noise scheduler 
    log.info(f"Instantiating joint diffusion scheduler <{cfg.diffusion_scheduler._target_}>")
    scheduler = hydra.utils.instantiate(cfg.diffusion_scheduler)

    # create transformation
    log.info(f"Instantiating latent transformation <{cfg.latent_transformation._target_}>")
    latent_transformation = hydra.utils.instantiate(cfg.latent_transformation).to(device)

    # create audio evaluator
    if cfg.audio_evaluator: 
        log.info(f"Instantiating audio evaluator <{cfg.audio_evaluator._target_}>")
        audio_evaluator = hydra.utils.instantiate(cfg.audio_evaluator).to(device)
    else:
        audio_evaluator = None

    if cfg.visual_evaluator:
        log.info(f"Instantiating visual evaluator <{cfg.visual_evaluator._target_}>")
        visual_evaluator = hydra.utils.instantiate(cfg.visual_evaluator).to(device)
    else:
        visual_evaluator = None

    log.info(f"Starting sampling!")
    clip_scores = []
    clap_scores = []
    
    for idx in tqdm(range(cfg.trainer.num_samples), desc='Sampling'):
        clip_score, clap_score = denoise(cfg, image_diffusion_guidance, audio_diffusion_guidance, scheduler, latent_transformation, visual_evaluator, audio_evaluator, idx, device)
        clip_scores.append(clip_score)
        clap_scores.append(clap_score)

    # re-ranking by metrics
    enable_rank = cfg.trainer.get("enable_rank", False)
    if enable_rank:
        log.info(f"Starting re-ranking and selection!")
        select_top_k_ranking(cfg, clip_scores, clap_scores)
    
    enable_clip_rank = cfg.trainer.get("enable_clip_rank", False)
    if enable_clip_rank:
        log.info(f"Starting re-ranking and selection by CLIP score!")
        select_top_k_clip_ranking(cfg, clip_scores)

    log.info(f"Finished!")


@torch.no_grad()
def denoise(cfg, image_diffusion, audio_diffusion, scheduler, latent_transformation, visual_evaluator, audio_evaluator, idx, device):
    image_guidance_scale, audio_guidance_scale = cfg.trainer.image_guidance_scale, cfg.trainer.audio_guidance_scale
    height, width = cfg.trainer.img_height, cfg.trainer.img_width
    image_start_step = cfg.trainer.get("image_start_step", 0)
    audio_start_step = cfg.trainer.get("audio_start_step", 0)
    audio_weight = cfg.trainer.get("audio_weight", 0.5)
    use_colormap = cfg.trainer.get("use_colormap", False)

    cutoff_latent = cfg.trainer.get("cutoff_latent", False)
    crop_image = cfg.trainer.get("crop_image", False)

    generator = torch.manual_seed(cfg.seed + idx)

    # obtain the text embeddings for each modality's diffusion process
    image_text_embeds = encode_prompt(cfg.trainer.image_prompt, image_diffusion, device, negative_prompt=cfg.trainer.image_neg_prompt, time_repeat=1)
    audio_file_path = cfg.trainer.get("audio_file_path", "")
    if audio_file_path:
        # disable text-based audio guidance when using external audio
        audio_text_embeds = None
    else:
        audio_text_embeds = encode_prompt(cfg.trainer.audio_prompt, audio_diffusion, device, negative_prompt=cfg.trainer.audio_neg_prompt, time_repeat=1)

    # if audio guidance is disabled (external audio), ensure image guidance starts immediately
    if audio_diffusion is None or audio_text_embeds is None:
        image_start_step = 0

    scheduler.set_timesteps(cfg.trainer.num_inference_steps)

    # init random latents
    latents = torch.randn((image_text_embeds.shape[0] // 2, image_diffusion.unet.config.in_channels, height // 8, width // 8), generator=generator, dtype=image_diffusion.precision_t).to(device)

    for i, t in enumerate(scheduler.timesteps):
        if i >= image_start_step: 
            image_noise = estimate_noise(image_diffusion, latents, t, image_text_embeds, image_guidance_scale)
        else: 
            image_noise = None
        
        if audio_text_embeds is not None and i >= audio_start_step:
            transform_latents = latent_transformation(latents, inverse=False)
            audio_noise = estimate_noise(audio_diffusion, transform_latents, t, audio_text_embeds, audio_guidance_scale)
            audio_noise = latent_transformation(audio_noise, inverse=True)
        else: 
            audio_noise = None
        
        if image_noise is not None and audio_noise is not None:
            noise_pred = (1.0 - audio_weight) * image_noise + audio_weight * audio_noise
        elif image_noise is not None and audio_noise is None:
            noise_pred = image_noise
        elif image_noise is None and audio_noise is not None:
            noise_pred = audio_noise
        else: 
            log.info("No estimated noise! Exit.")
            raise NotImplementedError
        
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)['prev_sample']

    if cutoff_latent and not crop_image:
        latents = latents[..., :-4] # we cut off 4 latents so that we can directly remove the black region

    # Img latents -> imgs
    img = image_diffusion.decode_latents(latents) # [1, 3, H, W]

    # Choose audio/spec source based on whether external file is provided
    audio_file_path = cfg.trainer.get("audio_file_path", "")
    if audio_file_path:
        # Load and prepare original audio
        audio_np, sr = sf.read(audio_file_path)
        if audio_np.ndim > 1:
            audio_np = librosa.to_mono(audio_np.T)
        if sr != 16000:
            audio_np = librosa.resample(audio_np.astype(float), orig_sr=sr, target_sr=16000)
        
        # Compute original audio spectrogram
        orig_spec = wav2spec(audio_np, 16000).to(device)  # [3, H, W]
        orig_spec_single = orig_spec.mean(dim=0, keepdim=True)  # [1, H, W]
        
        # Use generated grayscale image as a mask to modify the audio spectrogram
        # Convert image to grayscale and match spec dimensions
        img_gray = img.mean(dim=1, keepdim=True)  # [1, 1, H, W]
        img_gray = img_gray.squeeze(0)  # [1, H, W]
        
        # Resize image to match spectrogram dimensions if needed
        if img_gray.shape[-2:] != orig_spec_single.shape[-2:]:
            import torch.nn.functional as F
            img_gray = F.interpolate(img_gray.unsqueeze(0), size=orig_spec_single.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        
        # Apply imprint: use image as magnitude modulator
        mag_ratio = cfg.trainer.get("mag_ratio", 0.5)
        inverse_image = cfg.trainer.get("inverse_image", True)
        
        if inverse_image:
            img_gray = 1.0 - img_gray
        
        image_mask = 1 - mag_ratio * img_gray
        spec_modified = orig_spec_single * image_mask
        
        # Reconstruct audio from modified spectrogram using original phase
        audio = inverse_stft(spec_modified.cpu(), audio_np)
        audio = np.ravel(audio)
        
        # Use the modified spectrogram as the saved spec (convert back to 3-channel for consistency)
        spec = spec_modified.repeat(3, 1, 1)  # [3, H, W]
    else:
        # Standard path: decode spec from latents and convert to audio
        audio_latents = latent_transformation(latents, inverse=False)
        spec = audio_diffusion.decode_latents(audio_latents).squeeze(0) # [3, 256, 1024]
        audio = audio_diffusion.spec_to_audio(spec)
        audio = np.ravel(audio)

    if crop_image and not cutoff_latent:
        pixel = 32
        audio_length = int(pixel / width * audio.shape[0])
        img = img[..., :-pixel]
        spec = spec[..., :-pixel] 
        audio = audio[:-audio_length]   

    # evaluate with CLIP
    if visual_evaluator is not None:
        clip_score = visual_evaluator(img, cfg.trainer.image_prompt)
    else:
        clip_score = None

    # evaluate with CLAP
    if audio_evaluator is not None:
        clap_score = audio_evaluator(cfg.trainer.audio_prompt, audio)
    else:
        clap_score = None

    sample_dir = os.path.join(cfg.output_dir, 'results', f'example_{str(idx+1).zfill(3)}')
    os.makedirs(sample_dir, exist_ok=True)

    # save config with example-specific information 
    cfg_save_path = os.path.join(sample_dir, 'config.yaml')
    current_cfg = copy.deepcopy(cfg)
    current_cfg.seed = cfg.seed + idx
    with open_dict(current_cfg):
        current_cfg.clip_score = clip_score
        current_cfg.clap_score = clap_score
    OmegaConf.save(current_cfg, cfg_save_path)
    
    # save image
    img_save_path = os.path.join(sample_dir, 'img.png')
    save_image(img, img_save_path)

    # save audio
    audio_save_path = os.path.join(sample_dir, 'audio.wav')
    save_audio(audio, audio_save_path)

    # save spec (this is the one we want in the video)
    spec_raw_save_path = os.path.join(sample_dir, 'spec.png')
    save_image(spec.mean(dim=0, keepdim=True), spec_raw_save_path)

    # optional: save spec with colormap (separate file, not used for video)
    spec_colormap_path = None
    if use_colormap:
        spec_colormap_path = os.path.join(sample_dir, 'spec_colormap.png')
        spec_colormap = spec.mean(dim=0).cpu().numpy()
        plt.imsave(spec_colormap_path, spec_colormap, cmap='gray')

    # choose which spec to show in the video
    video_spec_path = spec_raw_save_path  # <-- always use spec.png

    # save video 
    video_output_path = os.path.join(sample_dir, 'video.mp4')
    if img.shape[-2:] == spec.shape[-2:]:
        create_single_image_animation_with_text(
            video_spec_path,
            audio_save_path,
            video_output_path,
            cfg.trainer.image_prompt,
            cfg.trainer.audio_prompt,
        )
    else:
        create_animation_with_text(
            img_save_path,
            video_spec_path,
            audio_save_path,
            video_output_path,
            cfg.trainer.image_prompt,
            cfg.trainer.audio_prompt,
        )

    
    return clip_score, clap_score


if __name__ == "__main__":
    main()
