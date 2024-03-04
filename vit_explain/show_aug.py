import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from model_utils.my_aug import AutoAugment, AutoAugmentPolicy
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


if __name__ == '__main__':
    args = get_args()
    args.use_cuda = True
    args.image_path = '/home/dataset/pendi/abnormal/V1350_2.jpg'
    # args.image_path = '/home/dataset/pendi/normal/V106_0.jpg'
    args.discard_ratio = 0.75
    # args.category_index = 281

    savePath = '../model_save/GP_75'
    model = torch.load(savePath + 'max')


    model.eval()

    if args.use_cuda:
        model = model.cuda()
    my_transform = transforms.Compose([
        # transforms.RandomResizedCrop(150),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        AutoAugment(policy=AutoAugmentPolicy.MY),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    # img = Image.open(args.image_path)
    img = cv2.imread(args.image_path)
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)
    plt.show()
    for i in range(10):
        input_tensor = my_transform(img)
        input_img = np.array(input_tensor)
        input_img = input_img.swapaxes(0, 1)
        input_img = input_img.swapaxes(1, 2)
        plt.imshow(input_img)
        plt.show()
