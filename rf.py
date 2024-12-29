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

from mmdit import MMDiT  # Ensure mmdit is correctly installed and accessible

# Define a transform to normalize the data to [-1, 1]
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5,), (0.5,)),
])

def train(
    batch_size=64,
    epochs=20,
    lr=1e-4,
    timesteps=50,
    device='cuda',
    save_model=True,
    log_wandb=False,
    download=True,
    save_dir='./models',
    dtype=torch.float32
):
    print(locals())

    # Initialize Weights & Biases
    if log_wandb:
        wandb.init(project='reflow-mnist', config=locals())

    device = torch.device(
        device if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    # Ensure the save directory exists
    if save_model:
        os.makedirs(save_dir, exist_ok=True)

    # Load the MNIST dataset
    mnist_train = datasets.MNIST(
        root='./datasets',
        train=True,
        transform=transform,
        download=download
    )

    train_loader = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the model and move it to the device
    model = MMDiT(dtype=dtype, device=device)
    model.train()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(device, dtype)  # Shape: (batch_size, 1, 28, 28)
            y = y.to(device, dtype)  # Shape: (batch_size,)

            # One-hot encode the labels
            y_onehot = F.one_hot(y, num_classes=10).float()

            # Sample x0 from a standard normal distribution
            x0 = torch.randn_like(x).to(device)

            optimizer.zero_grad()
            loss = 0.0

            for t in range(1, timesteps + 1):
                alpha = t / timesteps
                xt = alpha * x + (1 - alpha) * x0  # Linear interpolation

                t_tensor = torch.full((x.size(0),), t, dtype=dtype, device=device, requires_grad=False)

                vt = model(xt, t_tensor, y_onehot, y_onehot.clone())

                target = (x - x0) / timesteps  # Target for the velocity

                loss += F.mse_loss(vt, target)

            loss = loss / timesteps  # Average loss over timesteps
            loss.backward()
            optimizer.step()

            # Update progress bar and log metrics
            progress_bar.set_postfix({'loss': loss.item()})

            if log_wandb:
                wandb.log({'loss': loss.item(), 'epoch': epoch})

        # print(f'Epoch {epoch} Loss: {loss.item():.4f}')

        # Save model checkpoint every 5 epochs
        if save_model and epoch % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final model
    if save_model:
        final_model_path = os.path.join(save_dir, 'model_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model: {final_model_path}")

    if log_wandb:
        wandb.finish()


@torch.no_grad()
def sample_euler(
    num_steps=50,
    save_samples=True,
    sample_dir='./samples',
    model_path=None,
    device='cuda',
    create_gifs=True,
    gif_dir='./gifs',
    dtype=torch.float32
):

    device = torch.device(
        device if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    
    os.makedirs(sample_dir, exist_ok=True)
    if create_gifs:
        os.makedirs(gif_dir, exist_ok=True)


    # Initialize the model and load weights
    model = MMDiT(dtype=dtype, device=device)
    if model_path is None:
        raise ValueError("Please provide the path to the trained model checkpoint.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize a batch for all labels
    num_labels = 10  # MNIST has 10 classes
    x = torch.randn(num_labels, 1, 28, 28, dtype=dtype, device=device)  # Shape: (10, 1, 28, 28)
    labels = torch.arange(num_labels, device=device)  # Tensor([0,1,...,9])
    y = F.one_hot(labels, num_classes=10).to(dtype)  # Shape: (10, 10)

    # List to store images at each timestep for GIF creation
    images_per_label = [[] for _ in range(num_labels)]  # List of lists

    for t in tqdm(range(1, num_steps + 1), desc='Sampling Steps'):
        t_tensor = torch.full((num_labels,), t, dtype=dtype, device=device)  # Shape: (10,)

        vt = model(x, t_tensor, y, y.clone())  # Shape: (10, 1, 28, 28)
        x = x + vt  # Euler integration step

        if save_samples or create_gifs:
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

            # Optionally save intermediate images (optional)
            # Uncomment the following lines if you want to save images at each step
            # for idx in range(num_labels):
            #     save_path = os.path.join(sample_dir, f'label_{idx}_step_{t}.png')
            #     utils.save_image(x_denorm[idx], save_path)

    # Save the final images
    if save_samples:
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
