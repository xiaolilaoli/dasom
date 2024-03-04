import timm

print(timm.__version__)


import matplotlib.pyplot as plt


import torch

# a = torch.tensor([])
# b = torch.ones((12,4,6))
# for i in range(10):
#     a = torch.cat((a,b),dim=0)
# x = [0.5 + 0.05*i for i in range(9)]
# acc = [95.5, 95.5,96.0,97.0,97.5,96.5,95.5,95.5,95.5]
#
#
# plt.plot(x,acc)
#
# plt.ylim((90, 100))
# plt.xlim((0.5, 0.95))
# plt.xlabel('masking ratio')
# plt.ylabel('acc')
# plt.title('Acc and masking ratio')
# plt.legend(['ViT GroundPre'])
# plt.savefig('acc.png')
# plt.show()
#
# low = []
# mid = []
# hig = []
# k = 0
# for i in range(14):
#     for j in range(14):
#         if i < 2 or i > 11 or j<2 or j>11:
#             low.append(k)
#         elif i < 4 or i > 9 or j<4 or j>9:
#             mid.append(k)
#         else:
#             hig.append(k)
#         k+=1
# print(low)
# print(mid)
# print(hig)

# import matplotlib.pyplot as plt
#
# img = '/home/dataset/pendi/normal/V39_1.jpg'
# import cv2
# img = cv2.imread(img)
# plt.rcParams['figure.figsize'] = [10, 10]
#
# h, w, d = img.shape
# img = img[1 * h // 10:9 * h // 10, 1 * w // 10:8 * w // 10, :]
# img = cv2.resize(img, (224, 224))
# b, g, r = cv2.split(img)
# img = cv2.merge((r, g, b))
#
#
# plt.subplot(1, 5,1)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(1, 5, 2)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(1, 5, 3)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(1, 5, 4)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(1, 5, 5)
# plt.imshow(img)
# plt.axis('off')
# # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
# plt.show()

# f1 = [0.9207, 0.8974,0.9377,0.9276,0.9400,0.9450,0.9803]
# re = [0.9215,0.8998,0.9403,0.9302,0.9400,0.9450,0.9807]
#
#
# for i, eacn in enumerate(f1):
#     print(f1[i]*re[i]/(2*re[i]-f1[i]))
#
# print(0.9799*0.9807/(0.9799+0.9807)*2)

f1 = [0.6562,0.8853,0.7850, 0.9803]
re = [0.7472,0.8959,0.7850,0.9807]


for i, eacn in enumerate(f1):
    print(f1[i]*re[i]/(2*re[i]-f1[i]))









