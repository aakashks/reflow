import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from mmdit import MMDiT

from icecream import ic
ic.configureOutput(includeContext=True)

import lovely_tensors
lovely_tensors.monkey_patch()

# define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_train = datasets.MNIST(root='./datasets', train=True, transform=transform, download=False)
model = ic(MMDiT())


x = mnist_train.data[:2].unsqueeze(1) / 256
t = torch.tensor([0, 1])
y = mnist_train.targets[:2]

y = F.one_hot(y, 10).float()

out = ic(model(x, t, y, y.clone()))
