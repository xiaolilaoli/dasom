

from torch.utils.data import DataLoader,Dataset
import numpy as np

import torch
import csv

def testPendi(model,test_set):
    train_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    # 用测试集训练模型model(),用验证集作为测试集来验证
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    model.eval() # 确保 model_utils 是在 训练 model_utils (开启 Dropout 等...)
    for i, data in enumerate(train_loader):
        train_pred = model(data[0].cuda()) # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
        if data[1].item() == 0:
            if np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy():
                TN += 1
            else:
                FN += 1
        if data[1].item() == 1:
            if np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy():
                TP += 1
            else:
                FP += 1
    specificity = TP / (TP+FN)
    sensitivity = TN / (TN+FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('TN: %3.6f  FN: %3.6f TP: %3.6f FP: %3.6f ' % (TP, TN, FP, FN))
    print('acc: %3.6f spe: %3.6f sen: %3.6f'%(acc, specificity, sensitivity))

def evaluate(model_path, testloader, rel_path ,device):
    model = torch.load(model_path).to(device)
    # model = model_path
    # testloader = DataLoader(testset,batch_size=1,shuffle=False)  #放入loader 其实可能没必要 loader作用就是把数据形成批次而已
    test_rel = []
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(testloader):
            if i %1000 == 0:
                print(i)
            x = data.to(device)
            pred = model(x)
            idx = np.argmax(pred.cpu().data.numpy(),axis=1)
            for each in list(idx):
                test_rel.append(each)
    print(test_rel)
    with open(rel_path, 'w') as f:
        csv_writer = csv.writer(f)        #百度的csv写法
        csv_writer.writerow(['id','Class'])
        for i in range(len(testloader)):
            csv_writer.writerow([str(i),str(test_rel[i])])