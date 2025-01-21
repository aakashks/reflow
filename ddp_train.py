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
from loguru import logger
from omegaconf import OmegaConf

from model import DiT
from rectified_flow import RectifiedFlow
from sampling import sample

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = getattr(optim, config.training.get('optimizer', 'Adam'))
        
        self.batch_size = config.training.batch_size
        self.epochs = config.training.epochs
        self.logit_sampling = config.training.get('logit_sampling', False)
        self.device = config.training.device
        self.save_model = config.training.save_model
        self.wandb_offline = config.training.wandb_offline
        self.dataset_fraction = config.training.dataset_fraction
        self.project_name = f"{config.training.project_name}-{config.training.dataset}"
        self.dataset = config.training.dataset
        self.sample_examples = config.training.get('sample_examples', True)
        
        self.optimizer = self.optimizer(self.model.parameters(), lr=config.training.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.rf = RectifiedFlow(self.model)
        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        
        self.artifact_dir = './artifacts'  # Default artifact directory
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.train_loader = None

    def _init_wandb(self, rank=None):
        if wandb.run is not None:
            return

        if rank is not None and rank != 0:
            wandb.init(mode='disabled')
        else:
            config = OmegaConf.to_container(self.config, resolve=True)
            
            # convert to wandb friendly format
            cfg = {}
            for k, v in config.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        cfg[f'{k}.{kk}'] = vv
                else:
                    cfg[k] = v
            
            mode = 'online' if not self.wandb_offline else 'offline'
            logger.info('Train init', cfg)
            wandb.init(project=self.project_name, config=cfg, mode=mode)
            self.artifact_dir = wandb.run.dir
            os.makedirs(self.artifact_dir, exist_ok=True)

    def _get_dataloader(self, ddp=False, rank=0, world_size=1):
        if self.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = datasets.MNIST
        elif self.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = datasets.CIFAR10
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        dataset = dataset(root='./datasets', train=True, transform=transform, download=True)
        if self.dataset_fraction < 1.0:
            subset_len = int(len(dataset) * self.dataset_fraction)
            dataset = torch.utils.data.Subset(dataset, range(subset_len))

        if ddp:
            sampler = distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True        # ddp sampler shuffles no need to do it in dataloader
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
        # only on rank 0 process
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        checkpoint_path = os.path.join(self.artifact_dir, ckpt_name)
        torch.save(model.module.state_dict() if dist.is_initialized() else model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def _run_batch(self, x, y):
        x = x.to(self.device)  
        y = y.to(self.device)  

        self.optimizer.zero_grad()

        loss = self.rf.forward_pass(x, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _post_training(self):
        # only on rank 0 process do these things
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        if self.save_model:
            self._save_checkpoint(self.model, ckpt_name='model_final.pth')
            
        if self.sample_examples:
            sample(model=self.model, config=self.config, save_dir=self.artifact_dir)

        wandb.finish()
            
    def _pre_training(self, ddp, rank, world_size):
        if not ddp:
            self._init_wandb()  # Initialize wandb for non-DDP training
            # since no where else model is being moved for non-DDP case
            self.rf.model.to(self.device)
        
        if ddp:
            self.train_loader = self._get_dataloader(ddp=True, rank=rank, world_size=world_size)
        else:
            self.train_loader = self._get_dataloader(ddp=False)

        self.model.train()
        self.rf.model.to(self.device)
    
    def ddp_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        torch.cuda.set_device(rank)
        self.device = torch.device(f'cuda:{rank}')
        
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        self._init_wandb(rank)  # Initialize wandb with rank to reduce redundant inits

    def train(self, ddp=False, rank=0, world_size=1):
        self._pre_training(ddp, rank, world_size)
        
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epochs', disable=(ddp and rank != 0)):
            if ddp:
                # need to do for ddp
                self.train_loader.sampler.set_epoch(epoch)

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False, disable=(ddp and rank != 0))
            for batch_idx, (x, y) in enumerate(progress_bar):
                loss = self._run_batch(x, y)
                
                progress_bar.set_postfix({'loss': loss})    # tqdm 

                # only log on rank 0 process. however, only rank 0 loss will be logged
                # to help fix this one can use dist.all_reduce to get the average loss
                # TODO: dist.all_reduce
                if (not dist.is_initialized()) or (dist.get_rank() == 0):
                    wandb.log({
                        'loss': loss,
                        'epoch': epoch,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })

            self.scheduler.step()
            
            if self.save_model and epoch % 50 == 0:
                self._save_checkpoint(self.model, ckpt_name=f'model_epoch_{epoch}.pth')
            
        self._post_training()
        
        if rank == 0 or not ddp:
            logger.info("Training complete!")


def run_ddp(rank, world_size, config):
    try:
        model = DiT(**config.model)
        trainer = Trainer(model=model, config=config)
        trainer.ddp_setup(rank, world_size)
        trainer.train(ddp=True, rank=rank, world_size=world_size)
    except KeyboardInterrupt:
        logger.info(f"Process {rank} received keyboard interrupt")
    except Exception as e:
        logger.exception(f"Process {rank} failed with exception: {e}")
    finally:
        dist.destroy_process_group()


def train(config_path='configs/default.yaml'):
    config = OmegaConf.load(config_path)

    ddp = config.distributed.enabled
    world_size = config.distributed.world_size

    if not ddp or not torch.cuda.is_available() or torch.cuda.device_count() < 2 or world_size < 2 or config.training.device != 'cuda':
        logger.info("Training in non-DDP mode")
        model = DiT(**config.model)
        trainer = Trainer(model=model, config=config)
        trainer.train()
    else:
        mp.spawn(run_ddp, args=(world_size, config), nprocs=world_size)

if __name__ == '__main__':
    fire.Fire(train)
