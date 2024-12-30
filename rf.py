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


def train(
    batch_size=64,
    epochs=50,
    lr=5e-4,
    timesteps=25,
    device='cuda',
    save_model=True,
    wandb_offline=True,
    dataset_fraction=0.4,
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
    mnist_dataset = datasets.MNIST(
        root='./datasets',
        train=True,
        transform=transform,
        download=download
    )

    if dataset_fraction < 1.0:
        subset_len = int(len(mnist_dataset) * dataset_fraction)
        mnist_dataset = torch.utils.data.Subset(mnist_dataset, range(subset_len))


    train_loader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the model and move it to the device
    model = MMDiT().to(device)
    model.train()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device)  # Shape: (batch_size, 1, 28, 28)
            y = y.to(device)  # Shape: (batch_size,)

            # One-hot encode the labels
            y_onehot = F.one_hot(y, num_classes=10).float()

            # Sample x0 from a standard normal distribution
            x0 = torch.randn_like(x).to(device)

            optimizer.zero_grad()
            loss = 0.0

            for t in range(1, timesteps + 1):
                alpha = t / timesteps
                xt = alpha * x + (1 - alpha) * x0  # Linear interpolation

                t_tensor = torch.full((x.size(0),), t, device=device, requires_grad=False) / timesteps

                vt = model(xt, t_tensor, y_onehot, y_onehot.clone())

                target = (x - x0)  # Target for the velocity

                loss += F.mse_loss(vt, target)

            loss = loss / timesteps  # Average loss over timesteps
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
def sample_euler(
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
    x = torch.randn(num_labels, 1, 28, 28, device=device)  # Shape: (10, 1, 28, 28)
    labels = torch.arange(num_labels, device=device)  # Tensor([0,1,...,9])
    y = F.one_hot(labels, num_classes=10).float()  # Shape: (10, 10)

    # List to store images at each timestep for GIF creation
    images_per_label = [[] for _ in range(num_labels)]  # List of lists

    for t in tqdm(range(1, num_steps + 1), desc='Sampling Steps'):
        t_tensor = torch.full((num_labels,), t, device=device) / num_steps  # Shape: (10,)

        vt = model(x, t_tensor, y, y.clone())  # Shape: (10, 1, 28, 28)
        x = x + vt  # Euler integration step

        # Denormalize x to [0, 1] for saving
        x_denorm = (x * 0.5) + 0.5  # Shape: (10, 1, 28, 28)
        x_denorm = torch.clamp(x_denorm, 0, 1)

        # Append images for GIFs
        if create_gifs:
            for idx in range(num_labels):
                # Convert to CPU and NumPy for imageio
                img = x_denorm[idx].cpu().squeeze(0).numpy()  # Shape: (28, 28)
                img = (img * 255).astype('uint8')  # Convert to uint8
                images_per_label[idx].append(img)


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
        'sample': sample_euler
    })
