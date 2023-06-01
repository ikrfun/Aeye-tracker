import pandas as pd 
import numpy as np
import os 
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil

DATA_ROOT = os.path.normpath("D:\研究用データ\eye-conpe\data")
df_train = pd.read_csv(os.path.normpath("D:\研究用データ\eye-conpe\df_train.csv"))
df_val = pd.read_csv(os.path.normpath("D:\研究用データ\eye-conpe\df_val.csv"))

def make_dir():
#DATA_ROOTにあるデータのうち、df_trainにあるデータのみを抽出して/dataset/trainにコピー
    for i in range(len(df_train)):
        shutil.copy(os.path.join(DATA_ROOT, df_train.iloc[i, 0]), os.path.join("dataset/train", df_train.iloc[i, 0]))

    #DATA_ROOTにあるデータのうち、df_valにあるデータのみを抽出して/dataset/valにコピー
    for i in range(len(df_val)):
        shutil.copy(os.path.join(DATA_ROOT, df_val.iloc[i, 0]), os.path.join("dataset/val", df_val.iloc[i, 0]))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # モデルの入力サイズに合わせて画像をリサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
    ])

    train_dataset = ImageFolder(root='train_data_path', transform=transform)  # ここに訓練データのパスを指定
    val_dataset = ImageFolder(root='val_data_path', transform=transform)  # ここに検証データのパスを指定

    return train_dataset, val_dataset

traindataset, valdataset = make_dir()

if __name__ == "__main__":
    make_dir()