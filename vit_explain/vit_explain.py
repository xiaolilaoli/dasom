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
import timm.models.vision_transformer
from transformers import ViTFeatureExtractor, ViTForImageClassification
import os
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

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    args = get_args()
    args.use_cuda = True
    imglist = [
           '/home/dataset/pendi/abnormal/V780_2.jpg',
           '/home/dataset/pendi/abnormal/V973_0.jpg',
           '/home/dataset/pendi/normal/V110_4.jpg',
           '/home/dataset/pendi/abnormal/V54_2.jpg',
           '/home/dataset/pendi/abnormal/V69_1.jpg',
            '/home/dataset/pendi/normal/V106_0.jpg',
            '/home/dataset/pendi/abnormal/V1350_2.jpg'
               ]
    # ratio = [0.6, 0.6, 0.7, 0.6, 0.6]


    # # imglist = ['/home/dataset/pendi/abnormal/V101_0.jpg']
    # imglist = ['/home/dataset/pendi/normal/V103_3.jpg']
    H = 5
    ratio = [0.70, 0.6, 0.6, 0.6, 0.6]
    Wid = len(imglist)
    for i, image in enumerate(imglist):

        args.image_path = image
        # args.image_path = '/home/dataset/pendi/normal/V21_1.jpg'
        args.discard_ratio = ratio[i]
        cut = True
        # args.category_index = 281

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        savePath = '../model_save/max_auc/ViT'
        model = torch.load(savePath + 'max')

        plt.rcParams['figure.figsize'] = [10, 10]
        # show_image(x[0], "original")

        model.eval()

        if args.use_cuda:
            model = model.cuda()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        my_transform = transforms.Compose([
            # transforms.RandomResizedCrop(150),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            AutoAugment(policy=AutoAugmentPolicy.MY),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # img = Image.open(args.image_path)
        img = cv2.imread(args.image_path)



        h, w, d = img.shape
        if cut:
            img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8* w // 10, :]
        img = cv2.resize(img,(224, 224 ))

        plt.subplot(H, Wid, i+1)
        plt.imshow(img)
        plt.axis('off')
        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        # cv2.imshow("Input Image", np_img)

        b, g, r = cv2.split(mask)
        mask = cv2.merge((r, g, b))
        plt.subplot(H, Wid, Wid+1+i)
        plt.imshow(mask)
        plt.axis('off')
        # plt.savefig('../pic/explation_w', dpi=600, bbox_inches='tight')
        # plt.show()

        # cv2.imwrite("input.png", np_img)
        # plt.imwrite(name, mask)
        # cv2.waitKey(-1)

        #
        # cv2.imshow("Input Image", np_img)
        # cv2.imshow(name, mask)
        # cv2.imwrite("input.png", np_img)
        # cv2.imwrite(name, mask)
        # cv2.waitKey(-1)

        savePath = '../model_save/max_auc/WOGP_75'
        model = torch.load(savePath + 'max')


        model.eval()

        if args.use_cuda:
            model = model.cuda()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # img = Image.open(args.image_path)
        img = cv2.imread(args.image_path)
        h, w, d = img.shape
        if cut:
            img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8* w // 10, :]
        img = cv2.resize(img,(224, 224 ))

        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        # cv2.imshow("Input Image", np_img)

        b, g, r = cv2.split(mask)
        mask = cv2.merge((r, g, b))
        plt.subplot(H, Wid, Wid*2+1+i)
        plt.imshow(mask)
        # plt.savefig('../pic/explation_wo', dpi=600, bbox_inches='tight')
        # plt.show()
        plt.axis('off')


        # model= ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # model = timm.models.vit_base_patch16_224(pretrained=True)
        # savePath = '../model_save/max_auc/0.75'
        savePath = '../model_save/max_auc/0.75'

        model = torch.load(savePath + 'max')
        model.eval()

        if args.use_cuda:
            model = model.cuda()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # img = Image.open(args.image_path)
        img = cv2.imread(args.image_path)
        h, w, d = img.shape
        if cut:
            img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8* w // 10, :]
        img = cv2.resize(img,(224, 224 ))

        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        # cv2.imshow("Input Image", np_img)

        b, g, r = cv2.split(mask)
        mask = cv2.merge((r, g, b))
        plt.subplot(H, Wid, Wid*3+1+i)
        plt.imshow(mask)
        plt.axis('off')



        savePath = '../model_save/max_auc/0.5'

        model = torch.load(savePath + 'guidemax')
        model.eval()

        if args.use_cuda:
            model = model.cuda()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # img = Image.open(args.image_path)
        img = cv2.imread(args.image_path)
        h, w, d = img.shape
        if cut:
            img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8* w // 10, :]
        img = cv2.resize(img,(224, 224 ))

        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        # cv2.imshow("Input Image", np_img)

        b, g, r = cv2.split(mask)
        mask = cv2.merge((r, g, b))
        plt.subplot(H, Wid, Wid*4+1+i)
        plt.imshow(mask)
        plt.axis('off')



        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        plt.savefig('../pic/explation_vit', dpi=600, bbox_inches='tight')
        plt.show()

