
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import random
import matplotlib.pyplot as plt #画图
import cv2

HW = 224
#定义一个读取图片的函数readfile()
def readfile(path, label):
    # label 是一个布尔值，代表需不需要返回 y 值
    image_dir = sorted(os.listdir(path))
    # x存储图片，每张彩色图片都是128(高)*128(宽)*3(彩色三通道)
    x = np.zeros((len(image_dir), HW, HW, 3), dtype=np.uint8)
    # y存储标签，每个y大小为1
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
        x[i, :, :] = cv2.resize(img,(HW, HW))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

#分别将 training set、validation set、testing set 用函数 readfile() 读进来
workspace_dir = r'/home/dataset/food-11/food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

#training 时，通过随机旋转、水平翻转图片来进行数据增强（data augmentation）
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(), #随机翻转图片
    # transforms.RandomRotation(15), #随机旋转图片
    transforms.ToTensor(), #将图片变成 Tensor，并且把数值normalize到[0,1]
])

#testing 时，不需要进行数据增强（data augmentation）
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label 需要是 LongTensor 型
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)

trainloader = DataLoader(train_set,batch_size=16, shuffle=True)

valloader = DataLoader(val_set,batch_size=16, shuffle=True)