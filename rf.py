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

from mmdit import MMDiT 

class RF:
    def __init__(self, model):
        self.model = model
        
    def forward_pass(self, x, y, y0, time_steps, logit_sampling=False):
        loss = 0.0
        z1 = torch.randn_like(x, device=x.device, requires_grad=False)
        t_all = torch.rand(time_steps, x.size(0), requires_grad=False)    
        if logit_sampling:
            t_all = torch.randn_like(t_all)
            t_all = t_all * 20 - 10     # temporarily trying inverse logit sampling
            t_all = torch.sigmoid(t_all)

        # make it suitable for broadcasting
        t_all = t_all.view(time_steps, x.size(0), 1, 1, 1).to(x.device)

        zt = x * t_all + z1 * (1 - t_all)

        for t in range(0, time_steps):
            # TODO: add cfg
            t_all_ = t_all[t].squeeze()
            vt = self.model(zt[t], t_all_, y, y0)
            
            loss += F.mse_loss(vt, (x - z1))
            
        return loss / time_steps
    
    
    @torch.no_grad()
    def simple_euler(self, z, y, y0, steps=25):
        b = z.size(0)

        images = []

        dt = 1.0 / steps
        dt = torch.tensor([dt] * b, device=z.device, requires_grad=False)
        dt = dt.view(b, 1, 1, 1)
        
        for i in tqdm(range(0, steps), desc='Sampling Steps'):
            t = torch.tensor([i/steps] * b, device=z.device, requires_grad=False)
            
            vc = self.model(z, t, y, y0)
            
            z = z + dt*vc
            images.append(z.cpu())

        return images


def train(config_path='configs/default.yaml', **kwargs):
    config = OmegaConf.load(config_path)
    logger.info('Train init', config)

    # Initialize Weights & Biases
    mode = 'online' if not config.training.wandb_offline else 'offline'
    wandb.init(project=f"{config.training.project_name}-{config.training.dataset}", 
              config=OmegaConf.to_container(config, resolve=True), 
              mode=mode)

    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    artifact_dir = wandb.run.dir if wandb.run else './artifacts'
    os.makedirs(artifact_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load the dataset
    dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)

    if config.training.dataset_fraction < 1.0:
        subset_len = int(len(dataset) * config.training.dataset_fraction)
        dataset = torch.utils.data.Subset(dataset, range(subset_len))

    train_loader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # Initialize the model and move it to the device
    model = MMDiT(**config.model).to(device)
    model.train()
    
    rf = RF(model)

    optimizer = getattr(optim, config.training.get('optimizer', 'Adam'))(
        model.parameters(), 
        lr=config.training.lr
    )

    for epoch in tqdm(range(1, config.training.epochs + 1), desc='Epochs'):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.training.epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)  
            y = y.to(device)  

            y_onehot = F.one_hot(y, num_classes=config.model.num_classes).float()
            optimizer.zero_grad()

            loss = rf.forward_pass(x, y_onehot, y_onehot.clone(), 
                                   config.training.get('timesteps', 25), logit_sampling=config.training.get('logit_sampling', False))
            
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})
            wandb.log({'loss': loss.item(), 'epoch': epoch})

        if config.training.save_model and epoch % 5 == 0:
            checkpoint_path = os.path.join(artifact_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    if config.training.save_model:
        final_model_path = os.path.join(artifact_dir, 'model_final.pth')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model: {final_model_path}")

    wandb.finish()


@torch.no_grad()
def sample(
    sample_dir='./results',
    gif_dir='./results',
    num_steps=25,
    model_path=None,
    create_gifs=True,
    config_path='configs/default.yaml',
):
    config = OmegaConf.load(config_path)

    num_classes = config.model.get('num_classes', 10)
    
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(sample_dir, exist_ok=True)
    if create_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    model = MMDiT(**config.model).to(device)
    if model_path is None:
        raise ValueError("Please provide the path to the trained model checkpoint.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.randn(16, config.model.num_channels, 
                   config.model.input_size, config.model.input_size, device=device)
    labels = torch.arange(0, 16, device=device) % num_classes
    y = F.one_hot(labels, num_classes=num_classes).float()


    rf = RF(model)
    images_raw = rf.simple_euler(x, y, y.clone(), steps=num_steps)
    
    
    images_raw = torch.stack(images_raw)

    images_raw = images_raw * 0.5 + 0.5
    images_raw = torch.clamp(images_raw, 0, 1)

    # Save the last timestep image
    last_images = images_raw[-1]
    grid_img = utils.make_grid(last_images, nrow=4)
    save_path = os.path.join(sample_dir, 'grid.png')
    utils.save_image(grid_img, save_path)
    logger.info(f"Saved last timestep image at {save_path}")

    # Save the gif
    if create_gifs:
        gif_images = []
        for t in range(images_raw.size(0)):
            grid_img = utils.make_grid(images_raw[t], nrow=4)
            grid_img = (grid_img * 255).byte().permute(1, 2, 0).numpy()
            gif_images.append(grid_img)
        
        gif_path = os.path.join(gif_dir, 'grid.gif')
        imageio.mimsave(gif_path, gif_images, fps=5)
        logger.info(f"Saved GIF at {gif_path}")

    logger.info("Sampling complete.")

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'sample': sample
    })
