from torch.utils.data import Dataset
import torch
import tqdm
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from models.model import UNet
from dataset.sprites_dataset import SpritesDataset
import torchmetrics

# torch random normal
from diffusers import UNet2DModel
import pytorch_lightning as pl

transform = ToTensor()

device = "cpu"  # torch.device('mps')
model = UNet2DModel()


class DiffDataset(Dataset):
    def __init__(self):
        #self.dataset = torchvision.datasets.CIFAR10(
        #    root="cifar-10-batches-py", download=True, transform=transform
        #)
        #self.dataset = torchvision.datasets.MNIST("root", transform=ToTensor())
        self.dataset = SpritesDataset(transform, seed=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class DiffModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.model = UNet2DModel()
        #self.model = UNet(img_size=28, c_in=1, c_out=1, device="cuda")#UNet2DModel()
        self.model =  UNet(device="cuda")
        self.loss_fn = torch.nn.MSELoss()
        self.e = 0
        self.metric = torchmetrics.image.fid.FrechetInceptionDistance()
        self.T = 500
        self.beta_t = torch.linspace(1e-4, 0.02, self.T).to("cuda")
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)

    def forward(self, x, t):
        return self.model(x, t)#["sample"]

    def training_step(self, batch, batch_idx):
        out = self.forward_process(batch)
        x, y, t = out["img"], out["target"], out["t"]
        y_hat = self(x, t)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_real = batch
        x_real = (x_real * 255).type(torch.uint8)
        x_synth = self.sample()
        print(x_synth.shape)
        self.metric.update(x_real, real=True)
        print(x_real.shape)
        self.metric.update(x_synth, real=False)
        self.log("fid", self.metric)
    
    def on_validation_epoch_end(self):
        #self.metric.reset()
        self.sample(viz=True)
        
        
        
        
    
    def sample_step(self, x, t):
        eps_theta = self(x, t)
        z = torch.randn(x.size()).cuda()
        x = (1 / torch.sqrt(self.alpha[t])) * (x - ((1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])) * eps_theta) + torch.sqrt(self.beta[t]) * z
        return x
    
    def sample(self, viz=False):
        self.T = 500
        self.beta = torch.linspace(1e-4, 0.02, self.T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        x = torch.randn(256, 3, 16, 16).cuda()
        for t in range(self.T-1, 0, -1):
            x = self.sample_step(x, torch.tensor(t).cuda())
         
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        if viz:
            x_img = torchvision.utils.make_grid(x[:10], nrow=10)
            x_img = ToPILImage()(x_img)
            x_img.save(f"sample{self.e+1}.png")
            self.e += 1
        return x

    def forward_process(self, img):
        t = torch.randint(0, self.T, size=(img.shape[0],)).cuda()
        epsilon = torch.randn(img.size()).cuda()
        sample = (torch.sqrt(self.alpha_bar_t[t, None, None, None]) * img + torch.sqrt(1 - self.alpha_bar_t[t, None, None, None]) * epsilon)
        return {"img": sample, "target": epsilon, "t": t}
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dataset = DiffDataset()
    #breakpoint()
    # split into train and test
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    # define dataloader
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=3
    )
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=3
    )
    print(len(dataloader), len(val_dataloader))

    model = DiffModel()
    trainer = pl.Trainer(
        logger= pl.loggers.wandb.WandbLogger(),
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        callbacks=[pl.callbacks.ModelCheckpoint(
            monitor="fid",
            mode="min",
            save_top_k=1,
        )],
        limit_val_batches=1
    )  # Set gpus to the number of available GPUs
    #model = DiffModel.load_from_checkpoint("lightning_logs/ai5a4k80/checkpoints/epoch=87-step=13816.ckpt")
    #weights = torch.load("/zhome/ca/9/146686/exercise-3/models/weights-59epochs-full-dataset.pt")
    #model.model.load_state_dict(weights)
    trainer.fit(model, dataloader, val_dataloader)#, #ckpt_path="lightning_logs/1guf83kc/checkpoints/epoch=37-step=5966.ckpt")
    #model.eval()
    # set random seed
    #torch.manual_seed(1)
    #np.random.seed(1)
    #with torch.no_grad():
    #    model.sample()
