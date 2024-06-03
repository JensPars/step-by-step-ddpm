from torch.utils.data import Dataset
import torch
import tqdm 
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
# torch random normal
from diffusers import UNet2DModel
import pytorch_lightning as pl

transform = ToTensor()

device = "cpu" #torch.device('mps')
model = UNet2DModel()

class DiffDataset(Dataset):
    def __init__(self):
        self.dataset = torchvision.datasets.CIFAR10(root="cifar-10-batches-py", download=True, transform=transform)
        self.T = 1000
        self.beta_t = np.linspace(1e-4, 0.02, self.T)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = np.cumprod(self.alpha_t)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # sample t
        t = np.random.randint(0, self.T)
        # define the forward process    
        out = self.forward_process(item[0], t)
        return out
    
    def forward_process(self, img, t):
        epsilon = torch.randn(img.size())
        sample = np.sqrt(self.alpha_bar_t[t])*img + np.sqrt(1-self.alpha_bar_t[t])*epsilon
        return {"img": sample, "target": epsilon, "t": t}

class DiffModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, t):
        return self.model(x, t)['sample']

    def training_step(self, batch, batch_idx):
        x, y, t = batch["img"], batch["target"], batch["t"]
        y_hat = self(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, t = batch["img"], batch["target"], batch["t"]
        y_hat = self(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = DiffDataset()
    # split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # define dataloader
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = DiffModel()
    trainer = pl.Trainer()  # Set gpus to the number of available GPUs
    trainer.fit(model, dataloader, val_dataloader)
    
