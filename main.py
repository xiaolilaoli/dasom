import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae
sys.path.append('..')



# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    #刚才归一化了 现在返回 记得clip防止越界 int防止小数  因为像素都是整数   imshow竟然可以读张量
    plt.title(title, fontsize=16)

    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    print(model)
    return model
def plt_image(imgPath, model,mask_ratio=0.75):

    img = Image.open(imgPath)
    #raw是一种格式 stream 是确定能下再下。（比如会事先确定内存）
    img = img.resize((224, 224))
    # # image is [H, W, 3]
    # plt.imshow(img)
    # #刚才归一化了 现在返回 记得clip防止越界 int防止小数  因为像素都是整数   imshow竟然可以读张量
    # plt.title('1', fontsize=16)
    # plt.show()
    # plt.axis('off')
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]   #设置画布尺寸
    show_image(torch.tensor(img))
    plt.show()

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.5)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked_2 = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 3, 1)
    show_image(x[0], "original")



    plt.subplot(1, 3, 2)
    show_image(im_masked[0], "masked_50")

    plt.subplot(1, 3, 3)

    show_image(im_masked_2[0], "masked_75")

    plt.show()

import cv2

def run_one_image(imgPath, model, mask_ratio=0.75, cut=True):
    img = cv2.imread(imgPath)
    h, w, d = img.shape
    if cut:
        img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8 * w // 10, :]
    img = cv2.resize(img, (224, 224))

    # # image is [H, W, 3]
    # plt.imshow(img)
    # #刚才归一化了 现在返回 记得clip防止越界 int防止小数  因为像素都是整数   imshow竟然可以读张量
    # plt.title('1', fontsize=16)
    # plt.show()
    # plt.axis('off')
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    # plt.rcParams['figure.figsize'] = [10, 10]   #设置画布尺寸
    # show_image(torch.tensor(img))
    # plt.show()

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)


    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(3, 1, 1)
    # show_image(x[0], "original")
    show_image(x[0])

    plt.subplot(3, 1, 2)
    # show_image(im_masked[0], "masked")
    show_image(im_masked[0])
    # plt.subplot(4, 1, 3)
    # show_image(y[0], "reconstruction")
    plt.subplot(3, 1, 3)
    # show_image(im_paste[0], "reconstruction + visible")
    show_image(im_paste[0])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.05)
    plt.savefig('pic/maskandrec', dpi=600, bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots(3, 1, figsize=(24, 24))
    # for i in range(len(img)):
    #     ax[i].imshow(img[i])
    #     ax[i].set_xlabel("test csdn")
    # #如果要单独修改坐标轴
    # ax[5].set_xlabel("test csdn")
    # plt.show()






if __name__ == '__main__':
    # load an image
    # img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
    # # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    # img = Image.open(requests.get(img_url, stream=True).raw)
    imgpath = r'testPic/3.jpg'

    # chkpt_dir = 'mae_visualize_vit_large.pth'
    chkpt_dir = 'model_save/mae_visualize_vit_base.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
    print('Model loaded.')


    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(imgpath, model_mae)



