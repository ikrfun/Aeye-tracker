import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision.datasets import ImageFolder
import os 
from torchvision.models import resnet50
from preprocessing import get_transformers
from lightning.pytorch import Trainer
import lightning.pytorch as pl
#make_dataset function.
def get_dataset(dataset_path:str,transform:dict) -> tuple[Dataset]:
    # create the output directory if it does not exist
    train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform['train'])
    test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform['test'])

    return train_dataset,test_dataset


class Custom_Resnet(pl.LightningModule):
    def __init__(self,num_classes:int = 4):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        z = self.model(x)
        self.classifier(z)
        return z
        
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        #x = x.view(x.size(0),-1)
        z = self.model(x)
        loss = nn.CrossEntropyLoss()(z,y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x,y = batch
        #x = x.view(x.size(0),-1)
        z = self.model(x)
        val_loss = nn.CrossEntropyLoss(z,y)
        self.log('val_loss',val_loss)
    
    def test_step(self,batch,batch_idx):
        pass
    def predict_step(self,batch,batch_idx):
        pass

from argparse import ArgumentParser
from lightning.pytorch.callbacks import DeviceStatsMonitor

if __name__ == '__main__':
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("-i","--input", type=str, required=True)
    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    train_dataset,test_dataset = get_dataset(args.input,transform=get_transformers())
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

    # Use the parsed arguments in your program
    trainer = Trainer(
        profiler = 'advanced',
        callbacks=[DeviceStatsMonitor()],
        )
    model = Custom_Resnet()
    train_loader = DataLoader(train_set)
    trainer.fit(model = model, train_dataloaders = train_loader)

    
