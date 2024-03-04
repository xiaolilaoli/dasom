import cv2
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import shutil
from model_utils.my_aug import AutoAugment, AutoAugmentPolicy

HW = 224

def readjpgfile(listpath,label,rate = None,test=False):
    assert rate == None or rate//1 == rate
    # label 是一个布尔值，代表需不需要返回 y 值
    image_dir = sorted(os.listdir(listpath),key=lambda x: int(x[x.find('V')+1:x.find('_')]))
    n = len(image_dir)
    if rate:
        n = n*rate
    # x存储图片，每张彩色图片都是128(高)*128(宽)*3(彩色三通道)
    x = np.zeros((n, HW , HW , 3), dtype=np.uint8)
    # y存储标签，每个y大小为1
    if test:
        y = []
    else:
        y = np.zeros(n, dtype=np.uint8)
    if not rate:
        for i, file in enumerate(image_dir):
            # if i < 100:
            #     shutil.copy(os.path.join(listpath, file), os.path.join("/home/dataset/pendi/test/"+str(label), file))
            img = cv2.imread(os.path.join(listpath, file))
            h,w,d  = img.shape
            # cv2.imshow('file_ori', img)
            # cv2.waitKey(0)
            img = img[2*h//6:4*h//6, 2*w//6:4*w//6, :]
            # cv2.imshow('file', img)
            # cv2.waitKey(0)
            # xshape = img.shape0
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            x[i, :, :] = cv2.resize(img,(HW , HW ))
            if test:
                y.append(file)
            else:
                y[i] = label
    else:
        for i, file in enumerate(image_dir):
            img = cv2.imread(os.path.join(listpath, file))
            # xshape = img.shape
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            for j in range(rate):
                x[rate*i + j, :, :] = cv2.resize(img,(HW , HW ))
                y[rate*i + j] = label

    return x,y


#training 时，通过随机旋转、水平翻转图片来进行数据增强（data_abnor augmentation）
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(150),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(HW),
    AutoAugment(policy=AutoAugmentPolicy.MY),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#testing 时，不需要进行数据增强（data_abnor augmentation）
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

mae_trainform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class ImgDataset(Dataset):
    def __init__(self, x, y=None,file_path=None, transform=None):
        self.x = x
        # label 需要是 LongTensor 型
        self.y = y
        self.file_path = file_path
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
            return X,self.file_path[index]
    def getbatch(self,indices):
        images = []
        labels = []
        for index in indices:
            image,label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images),torch.tensor(labels)



def getDateset(dir_class1, dir_class2=None, testSize=0.3,rate = None, testNum = None, trainsform=None, lessTran = False):
    '''
    :param dir_class1:   这个是参数较少的那个
    :param dir_class2:
    :param testSize:
    :param rate:
    :param testNum:
    :return:
    '''
    #类2是1
    if testNum == -1:
        x1,y1 = readjpgfile(dir_class1,0,rate=rate)
        dataset = ImgDataset(x1, y1, transform=train_transform)
        return dataset
    x1,y1 = readjpgfile(dir_class1,0,rate=rate)  #类1是0
    x2,y2 = readjpgfile(dir_class2,1)
    if not testNum :
        X = np.concatenate((x1, x2))
        Y = np.concatenate((y1, y2))
        train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=testSize,random_state=0)

    else:
        # train_x1 = x1[:-testNum][:20]
        # test_x1 = x1[-testNum:]
        # train_y1 = y1[:-testNum][:20]
        # test_y1 = y1[-testNum:]
        # train_x2 = x2[:-testNum][:20]
        # test_x2 = x2[-testNum:]
        # train_y2 = y2[:-testNum][:20]
        # test_y2 = y2[-testNum:]


        train_x1 = x1[testNum:]
        test_x1 = x1[:testNum]
        train_y1 = y1[testNum:]
        test_y1 = y1[:testNum]
        train_x2 = x2[testNum:]
        test_x2 = x2[:testNum]
        train_y2 = y2[testNum:]
        test_y2 = y2[:testNum]

        train_x1 = x1[testNum:]
        test_x1 = x1[:testNum]
        train_y1 = y1[testNum:]
        test_y1 = y1[:testNum]
        train_x2 = x2[testNum:]
        test_x2 = x2[:testNum]
        train_y2 = y2[testNum:]
        test_y2 = y2[:testNum]

        # train_x1 = x1[testNum:][:40]
        # test_x1 = x1[:testNum]
        # train_y1 = y1[testNum:][:40]
        # test_y1 = y1[:testNum]
        # train_x2 = x2[testNum:][:40]
        # test_x2 = x2[:testNum]
        # train_y2 = y2[testNum:][:40]
        # test_y2 = y2[:testNum]
        ratio = len(train_x1)//len(train_x2)
        train_x2 = np.array([train_x2[i // ratio] for i in range(len(train_x2) * ratio)])
        train_y2 = np.array([train_y2[i // ratio] for i in range(len(train_y2) * ratio)])



        # train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y1,test_size=testNum/len(y2),random_state=0)
        # train_x2, test_x2, train_y2, test_y2 = train_test_split(x2,y2,test_size=testNum/len(y2),random_state=0)
        print(len(test_y2),len(test_y1))
        # train_num = 20
        # train_x = np.concatenate((train_x1[:train_num],train_x2[:train_num]))
        # test_x = np.concatenate((test_x1, test_x2))
        # train_y = np.concatenate((train_y1[:train_num],train_y2[:train_num]))
        # test_y = np.concatenate((test_y1, test_y2))

        train_x = np.concatenate((train_x1,train_x2))
        test_x = np.concatenate((test_x1, test_x2))
        train_y = np.concatenate((train_y1,train_y2))
        test_y = np.concatenate((test_y1, test_y2))

        train_x_mae = np.concatenate((train_x1,train_x2))
        test_x = np.concatenate((test_x1, test_x2))
        train_y_mae = np.concatenate((train_y1,train_y2))
        test_y = np.concatenate((test_y1, test_y2))
        # train_x = train_x_mae
        # train_y = train_y_mae

    train_dataset = ImgDataset(train_x,train_y ,transform=train_transform)
    test_dataset = ImgDataset(test_x ,test_y,transform=test_transform)
    # mae_dataset = ImgDataset(train_x_mae,train_y_mae,transform=mae_trainform)
    mae_dataset = ImgDataset(train_x_mae, train_y_mae, transform=mae_trainform)


    return train_dataset, test_dataset,mae_dataset

def get_test_dataset(dir_class):

    #类2是1
    x1,y1 = readjpgfile(dir_class,0,rate=None,test=True)
    dataset = ImgDataset(x1, file_path=y1, transform=test_transform)
    return dataset





