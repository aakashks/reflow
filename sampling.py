import os
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import wandb
import imageio
import fire
from omegaconf import OmegaConf
from loguru import logger

from model import DiT
from rectified_flow import RectifiedFlow


@torch.inference_mode()
def sample(
    save_dir='./results',
    num_steps=20,
    null_cond=True,
    cfg=2.0,
    model=None,
    model_path=None,
    config=None,
    save_name='grid',
    config_path='configs/mnist.yaml',
):
    config = OmegaConf.load(config_path) if config is None else config

    num_classes = config.model.get('num_classes', 10)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)

    if model is None:
        model = DiT(**config.model).to(device)
        if model_path is None:
            raise ValueError("Please provide the path to the trained model checkpoint.")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    else:
        model.to(device)
    model.eval()

    x = torch.randn(16, config.model.num_channels, 
                   config.model.input_size, config.model.input_size, device=device)
    cond = torch.arange(0, 16, device=device) % num_classes
    null_cond = torch.full_like(cond, num_classes) if null_cond else None

    rf = RectifiedFlow(model)
    images_raw = rf.simple_euler(x, cond, null_cond, cfg=cfg, steps=num_steps)
    
    images_raw = torch.stack(images_raw)

    images_raw = images_raw * 0.5 + 0.5
    images_raw = torch.clamp(images_raw, 0, 1)

    # Save the last timestep image
    last_images = images_raw[-1]
    grid_img = utils.make_grid(last_images, nrow=4)
    save_path = os.path.join(save_dir, f'{save_name}.png')
    utils.save_image(grid_img, save_path)
    logger.info(f"Saved last timestep image at {save_path}")

    # Save the gif
    gif_images = []
    for t in range(images_raw.size(0)):
        grid_img = utils.make_grid(images_raw[t], nrow=4)
        grid_img = (grid_img * 255).byte().permute(1, 2, 0).numpy()
        gif_images.append(grid_img)
    
    gif_path = os.path.join(save_dir, f'{save_name}.gif')
    imageio.mimsave(gif_path, gif_images, fps=5)
    logger.info(f"Saved GIF at {gif_path}")

    logger.info("Sampling complete.")
    
if __name__ == '__main__':
    fire.Fire(sample)
