import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import os 
from torchvision.models import resnet50
from preprocessing import get_transformers
from lightning.pytorch import Trainer




#make_dataset function.
def get_dataset(dataset_path:str,transform:dict) -> tuple[Dataset]:
    # create the output directory if it does not exist
    train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform['train'])
    test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform['test'])

    return train_dataset,test_dataset


class Custom_Resnet(pl.LightningModule):
    def __init__(self,num_classes:int = 4):
        super().__init__()
        self.save_hyperparameters()
        backbone = resnet50(weights = "DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters,num_classes)

    def setup(self):
        train_dataset,test_dataset = get_dataset(DATASET_DIR,transform=get_transformers())
        train_set_size = int(len(train_dataset) * 0.8)
        valid_set_size = len(train_dataset) - train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    def forward(self, batch, batch_idx):
        x,y = batch
        
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        #x = x.view(x.size(0),-1)
        z = self.feature_extractor(x)
        z = self.classifier(z)
        loss = nn.CrossEntropyLoss(z,y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x,y = batch
        #x = x.view(x.size(0),-1)
        z = self.feature_extractor(x)
        z = self.classifier(z)
        val_loss = nn.CrossEntropyLoss(z,y)
        self.log('val_loss',val_loss)
    
    def test_step(self,batch,batch_idx):
        
    def predict_step(self,batch,batch_idx):
        

from argparse import _ArgumentParser
from lightning.pytorch.callbacks import DeviceStatsMonitor

if __name__ == '__main__':
    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--devices", type=int, default=2)
    # Hyperparameters for the model
    parser.add_argument("--layer_1_dim", type=int, default=128)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    # Use the parsed arguments in your program
    trainer = Trainer(
        profiler = 'advanced',
        callbacks=[DeviceStatsMonitor()],

        )

    
