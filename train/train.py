import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchvision.models import resnet34
from preprocessing import get_mean_std

#make_dataset function.
def get_dataset(dataset_path:str,transform:dict) -> tuple[Dataset]:
    # create the output directory if it does not exist
    train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform['train'])
    test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform['test'])

    return train_dataset,test_dataset

class PlDataModule(pl.LightningDataModule):

    def __init__(self, dataset_train, dataset_val, dataset_test, batch_size: int = 512):
        super().__init__()
        
        self.batch_size = batch_size
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

    def setup(self,stage:str = None):
        if stage == 'fit' or stage is None:
        self.train_set = MyDataset(
            self.train_dir,
            transform=self.data_augmentation
        )
        size = len(self.train_set)
        t, v = (int(size * 0.9), int(size * 0.1)) # if using holdout method
        t += (t + v != size)
        self.train_set, self.valid_set = random_split(self.train_set, [t, v])

    if stage == 'test' or stage is None:
        self.test_set = MyDataset(
            self.test_dir,
            transform=self.transform
        )

    def train_dataloader(self):
        return 

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size
        )

class PlResnetModule(pl.LightningModule):
    def __init__(self, lr=0.001, classes=4):
        super().__init__()
        self.lr = lr
        self.classes = classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = resnet34(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        return optimizer

def train(mean_std_dir, dataset_dir, epochs):
    # データの前処理
    transformer = get_transformers(get_mean_std(image_dir_path=mean_std_dir), (224, 224))
    dataset_train, dataset_val, dataset_test = get_dataset(transformer, dataset_dir)

    # モデルとデータセットを使って PyTorch Lightning モデルを作成する
    resnet_module = PlResnetModule()
    pl_data_module = PlDataModule(dataset_train, dataset_val, dataset_test)

    # PyTorch Lightning の Trainer オブジェクトを作成する
    trainer = pl.Trainer(
        tpu_cores = 8,
        max_epochs=epochs,
        progress_bar_refresh_rate=10,
        gpus=torch.cuda.device_count()
    )

    # モデルを訓練する
    trainer.fit(resnet_module, pl_data_module)
    # テストデータでモデルを評価する
    result = trainer.test(resnet_module, datamodule=pl_data_module)
    # テスト結果を出力する
    print(result)
    # モデルを保存する
    torch.save(resnet_module.model.state_dict(), 'resnet_model.pth')

