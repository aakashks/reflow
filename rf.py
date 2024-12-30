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

from mmdit import MMDiT 

class RF:
    def __init__(self, model):
        self.model = model
        
    def forward_pass(self, x, y, y0, time_steps):
        loss = 0.0
        z1 = torch.randn_like(x, device=x.device, requires_grad=False)
        
        for t in range(1, time_steps + 1):
            
            t = t / time_steps
            zt = x * t + z1 * (1 - t)
            
            t = torch.full((x.size(0),), t, device=x.device, requires_grad=False) / time_steps
            vt = self.model(zt, t, y, y0)
            
            loss += F.mse_loss(vt, (x - z1))
            
        return loss / time_steps
    
    
    @torch.no_grad()
    def simple_euler(self, x, y, y0, steps=25):
        b = x.size(0)

        images_per_label = [[] for _ in range(b)]

        for t in tqdm(range(1, steps + 1), desc='Sampling Steps'):
            t_tensor = torch.full((b,), t, device=x.device) / steps

            vt = self.model(x, t_tensor, y, y0)
            x = x + vt

            x_denorm = (x * 0.5) + 0.5
            x_denorm = torch.clamp(x_denorm, 0, 1)

            for idx in range(b):
                img = x_denorm[idx].cpu().squeeze(0).numpy()
                img = (img * 255).astype('uint8')
                images_per_label[idx].append(img)

        return images_per_label


def train(
    batch_size=64,
    epochs=50,
    lr=5e-4,
    timesteps=25,
    device='cuda',
    save_model=True,
    wandb_offline=False,
    dataset_fraction=0.2,
    download=True,
):
    print(locals())

    # Initialize Weights & Biases
    if not wandb_offline:
        wandb.init(project='reflow-mnist', config=locals())
    else:
        wandb.init(project='reflow-mnist', config=locals(), mode='offline')

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    artifact_dir = wandb.run.dir if wandb.run else './artifacts'
    os.makedirs(artifact_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load the MNIST dataset
    dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=download)

    if dataset_fraction < 1.0:
        subset_len = int(len(dataset) * dataset_fraction)
        dataset = torch.utils.data.Subset(dataset, range(subset_len))


    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize the model and move it to the device
    model = MMDiT().to(device)
    model.train()
    
    rf = RF(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)  
            y = y.to(device)  

            # One-hot encode the labels
            y_onehot = F.one_hot(y, num_classes=10).float()
            optimizer.zero_grad()

            loss = rf.forward_pass(x, y_onehot, y_onehot.clone(), timesteps)
            
            loss.backward()
            optimizer.step()

            # Update progress bar and log metrics
            progress_bar.set_postfix({'loss': loss.item()})

            wandb.log({'loss': loss.item(), 'epoch': epoch})

        # print(f'Epoch {epoch} Loss: {loss.item():.4f}')

        # Save model checkpoint every 5 epochs
        if save_model and epoch % 5 == 0:
            checkpoint_path = os.path.join(artifact_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final model
    if save_model:
        final_model_path = os.path.join(artifact_dir, 'model_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model: {final_model_path}")


    wandb.finish()


@torch.no_grad()
def sample(
    num_steps=25,
    sample_dir='./samples',
    model_path=None,
    device='cuda',
    create_gifs=True,
    gif_dir='./gifs',
    num_labels=10,
):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(sample_dir, exist_ok=True)
    if create_gifs:
        os.makedirs(gif_dir, exist_ok=True)


    # Initialize the model and load weights
    model = MMDiT().to(device)
    if model_path is None:
        raise ValueError("Please provide the path to the trained model checkpoint.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize a batch for all labels
    x = torch.randn(num_labels, 1, 28, 28, device=device)
    labels = torch.arange(num_labels, device=device)
    y = F.one_hot(labels, num_classes=10).float()

    rf = RF(model)
    
    images_per_label = rf.simple_euler(x, y, y.clone(), steps=num_steps)

    # Save the final images
    for idx in range(num_labels):
        save_path = os.path.join(sample_dir, f'label_{idx}.png')
        imageio.imwrite(save_path, images_per_label[idx][-1])
        print(f"Saved sample for label {idx} at {save_path}")

    # Create and save GIFs
    if create_gifs:
        for idx in range(num_labels):
            gif_path = os.path.join(gif_dir, f'label_{idx}.gif')
            imageio.mimsave(gif_path, images_per_label[idx], fps=5)
            print(f"Saved GIF for label {idx} at {gif_path}")

    print("Sampling complete.")


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'sample': sample
    })
