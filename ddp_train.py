import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import wandb
import fire

from mmdit import MMDiT 
from rf import RF

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        batch_size=64,
        epochs=50,
        lr=5e-4,
        timesteps=25,
        device='cuda',
        save_model=True,
        wandb_offline=False,
        dataset_fraction=0.2,
        project_name='reflow-mnist',
    ):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.timesteps = timesteps
        self.device = device
        self.save_model = save_model
        self.wandb_offline = wandb_offline
        self.dataset_fraction = dataset_fraction
        self.project_name = project_name  # Store project_name
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        config = {'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'timesteps': timesteps, 'dataset_fraction': dataset_fraction}

        self.rf = RF(model)
        
        self.config = config  # Store config for later wandb init
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.artifact_dir = './artifacts'  # Default artifact directory
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.train_loader = None

    def _init_wandb(self, rank=None):
        if wandb.run is not None:
            return

        if rank is not None and rank != 0:
            wandb.init(mode='disabled')
        else:
            mode = 'online' if not self.wandb_offline else 'offline'
            print('Train init', self.config)
            wandb.init(project=self.project_name, config=self.config, mode=mode)
            self.artifact_dir = wandb.run.dir
            os.makedirs(self.artifact_dir, exist_ok=True)

    def _get_dataloader(self, ddp=False, rank=0, world_size=1):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)

        if self.dataset_fraction < 1.0:
            subset_len = int(len(dataset) * self.dataset_fraction)
            dataset = torch.utils.data.Subset(dataset, range(subset_len))

        if ddp:
            sampler = distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                sampler=sampler
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

        return train_loader
    
    def _save_checkpoint(self, model, ckpt_name='model_final.pth'):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        checkpoint_path = os.path.join(self.artifact_dir, ckpt_name)
        torch.save(model.module.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
    def _run_batch(self, x, y):
        x = x.to(self.device)  
        y = y.to(self.device)  

        y_onehot = F.one_hot(y, num_classes=10).float()
        self.optimizer.zero_grad()

        loss = self.rf.forward_pass(x, y_onehot, y_onehot.clone(), self.timesteps)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _post_training(self):
        if self.save_model:
            self._save_checkpoint(self.model, ckpt_name='model_final.pth')

        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.finish()
    
    def ddp_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
        
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        self._init_wandb(rank)  # Initialize wandb with rank

    def train(self, ddp=False, rank=0, world_size=1):
        if not ddp:
            self._init_wandb()  # Initialize wandb for non-DDP training
            self.rf.model.to(self.device)
        
        if ddp:
            self.train_loader = self._get_dataloader(ddp=True, rank=rank, world_size=world_size)
        else:
            self.train_loader = self._get_dataloader(ddp=False)

        self.model.train()
        
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epochs', disable=(ddp and rank != 0)):
            if ddp:
                self.train_loader.sampler.set_epoch(epoch)

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False, disable=(ddp and rank != 0))
            for batch_idx, (x, y) in enumerate(progress_bar):
                loss = self._run_batch(x, y)
                
                progress_bar.set_postfix({'loss': loss})

                if (not dist.is_initialized()) or (dist.get_rank() == 0):
                    wandb.log({
                        'loss': loss,
                        'epoch': epoch,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })

            self.scheduler.step()
            
            if self.save_model and epoch % 5 == 0:
                self._save_checkpoint(self.model, ckpt_name=f'model_epoch_{epoch}.pth')
            
        self._post_training()
        
        if rank == 0 or not ddp:
            print("Training complete!")


def run_ddp(rank, world_size, kwargs):
    model = MMDiT()
    trainer = Trainer(
        model=model,
        optimizer=optim.Adam,
        **kwargs
    )
    trainer.ddp_setup(rank, world_size)
    trainer.train(ddp=True, rank=rank, world_size=world_size)
    dist.destroy_process_group()


def train_ddp(**kwargs):
    not_ddp = kwargs.pop('not_ddp', False)
    world_size = kwargs.pop('world_size', 2)
    
    if world_size == 1 or not_ddp:
        print("Training in non-DDP mode")
        model = MMDiT()
        trainer = Trainer(
            model=model,
            optimizer=optim.Adam,
            **kwargs
        )
        trainer.train()
        return
    
    mp.spawn(run_ddp, args=(world_size, kwargs), nprocs=world_size)


if __name__ == '__main__':
    fire.Fire(train_ddp)
