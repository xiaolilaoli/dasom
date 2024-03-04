import os
import cv2
from shutil import copy
import shutil


def getFileList(dir, Filelist,ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)

    return Filelist

def pro(index):
    imgPath = pathClass1 + '/' + last[index]
    img = cv2.imread(imgPath)
    img = cv2.resize(img,(720,720))
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('1'):
        shutil.move(imgPath, movepath1 + '/'+ each)
    elif key == ord('2'):
        shutil.move(imgPath, movepath2 + '/'+ each)
    elif key == ord('b'):
        try:
            pro(index-1)
        except:
            pro(index-2)






pathClass1 = r'E:\dataset\huaxi\dataset'
# pathClass2 = r'F:\pycharm\deepLeaning\pendi_cv2\data_abnor\val'

normalpath = r'E:\dataset\pendi\normal'
abnormalpath = r'E:\dataset\pendi\abnormal'

# nor_file_paths = []
# abnor_file_paths = []
# class1dir = os.listdir(pathClass1)
# for dir_0 in class1dir:
#     dir_1 = os.path.join(pathClass1,dir_0)
#     if os.path.isdir(dir_1):
#         for file_path in os.listdir(dir_1):
#             if dir_0[3] == '正':
#                 nor_file_paths.append(os.path.join(dir_1,file_path))
#             else:
#                 abnor_file_paths.append(os.path.join(dir_1,file_path))
#
# for i,file_path in enumerate(nor_file_paths):
#     pic_names = os.listdir(file_path)
#     index = 0
#     for pic in pic_names:
#         if 'V' in pic:
#             pic_path = os.path.join(file_path,pic)
#             new_path = normalpath + '/V' + str(i)+f'_{index}.jpg'
#             index += 1
#             shutil.copy(pic_path, new_path)

# for i,file_path in enumerate(abnor_file_paths):
#     pic_names = os.listdir(file_path)
#     index = 0
#     for pic in pic_names:
#         if 'V' in pic:
#             pic_path = os.path.join(file_path,pic)
#             new_path = abnormalpath + '/V' + str(i)+f'_{index}.jpg'
#             index += 1
#             shutil.copy(pic_path, new_path)
#
# line_nor_path = r'E:\dataset\pendi\line_nor'
# line_abn_path = r'E:\dataset\pendi\line_abn'
# noLine_nor_path = r'E:\dataset\pendi\noLine_nor'
# noLine_abn_path = r'E:\dataset\pendi\noLine_abn'
# class2dir = os.listdir(pathClass2)
img_path = "/home/dataset/pendi/test/atten/0"
now_path = "/home/dataset/pendi/test/atten/abN"
i = 0
last = []
index = 0
nor_pics = os.listdir(img_path)
for pic in nor_pics:
    pic_path = os.path.join(img_path,pic)
    img = cv2.imread(pic_path)
    img = cv2.resize(img,(720,720))
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('1'):
        shutil.copy(pic_path, os.path.join(now_path,pic))


# for each in class1dir[i:]:
#     imgPath = pathClass1 + '/' + each
#     img = cv2.imread(imgPath)
#     img = cv2.resize(img,(720,720))
#     cv2.imshow('img', img)
#     key = cv2.waitKey(0)
#     if key == ord('1'):
#         shutil.move(imgPath, movepath1 + '/' + each)
#     elif key == ord('2'):
#         shutil.move(imgPath, movepath2 + '/' + each)
#     elif key == ord('b'):
#         pro(index-1)
#     else:
#         last.append(each)
#
#     i += 1
#     print(i)


