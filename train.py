import torch
from torch.utils.data import DataLoader
from model import Model
from prepare import traindataset, valdataset

# データセットとデータローダーの設定


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# モデル、最適化手法、損失関数の設定
model = Model().cuda()  # GPUを使用
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ここではAdamを使用

# 訓練ループ
num_epochs = 10  # エポック数
for epoch in range(num_epochs):
    model.train()  # モデルを訓練モードに
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()  # GPUを使用
        labels = labels.cuda()  # GPUを使用

        optimizer.zero_grad()  # 勾配をゼロに
        loss = model.loss({"face": inputs}, labels)  # 損失を計算
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータを更新

    model.eval()  # モデルを評価モードに
    with torch.no_grad():  # 勾配計算を無効に
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()  # GPUを使用
            labels = labels.cuda()  # GPUを使用

            loss = model.loss({"face": inputs}, labels)  # 損失を計算
            # ここで損失を記録したり、他の評価指標を計算したりする
