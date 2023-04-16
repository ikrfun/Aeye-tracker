

from dataclasses import dataclass
import random
import os
import random
import shutil
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def get_mean_std(image_dir_path: str,max_images:int = 500) -> tuple:
    images = []
    
    for filename in os.listdir(image_dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(image_dir_path, filename)
            with Image.open(filepath) as img:
                # 画像を numpy 配列に変換して images リストに追加
                images.append(np.asarray(img))
                img.close()
                if len(images) >= max_images:
                    break

    # images リストを numpy 配列に変換
    images = np.stack(images)
    # 平均と標準偏差を計算
    mean = np.mean(images, axis=(0, 1, 2)) / 255.0
    std = np.std(images, axis=(0, 1, 2)) / 255.0
    # 結果をタプルで返す
    print(f'mean:{mean}')
    print(f'std:{std}')
    return mean,std

import random
import torchvision.transforms as transforms

class RandomExposure(object):
    """ランダムな露出度を加える前処理"""

    def __init__(self, min_gamma=0.5, max_gamma=2.0):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, x):
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        x = transforms.functional.adjust_gamma(x, gamma)
        return x

class RandomBrightness(object):
    """ランダムな明るさを加える前処理"""

    def __init__(self, min_brightness=-0.1, max_brightness=0.1):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def __call__(self, x):
        brightness = random.uniform(self.min_brightness, self.max_brightness)
        x = transforms.functional.adjust_brightness(x, brightness)
        return x

class RandomNoise(object):
    """ランダムなノイズを加える前処理"""

    def __init__(self, min_noise=0.0, max_noise=0.05):
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, x):
        noise = random.uniform(self.min_noise, self.max_noise)
        x = transforms.functional.adjust_hue(x, noise)
        return x

#画像前処理用のtransformerを作成する
def get_transformers(image_dir:str = 'dataset/train/0', image_size:int = 224)-> dict[str, transforms.Compose]:
    mean,std = get_mean_std(os.path.normpath(image_dir))
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([RandomExposure()], p=0.5),
            transforms.RandomApply([RandomBrightness()], p=0.5),
            transforms.RandomApply([RandomNoise()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ]),
        'test':transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    }
    return data_transform





