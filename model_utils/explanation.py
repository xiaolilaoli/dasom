import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.segmentation import slic
import lime
from lime import lime_image
from data import getDateset

def get_k_layer_feature_map(model_layer, k, x):
    x = x.cuda()
    with torch.no_grad():
        for index, layer in enumerate(model_layer):# model的第一个Sequential()是有多层，所以遍历

            x = layer(x)                           # torch.Size([1, 64, W, H])生成了64个通道
            if k == index:
                return x
def hook(model,input,output):
    print('hooking')
    global layer_activations
    layer_activations=output

def run_layer(model_layer, x):
    with torch.no_grad():
        x = model_layer(x)
        # [1, 64, 112, 112]
    return x

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

#画出整张图的梯度大小
def pltimportant(model,img_indices, train_dataset):

    model.eval()
    x, y= train_dataset.getbatch(img_indices)
    model = model.to('cpu')
    x.requires_grad_()   #为x赋值requires_grad为True
    y_pred=model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss=loss_func(y_pred,y)
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])   #这步只是为了标准化

    x=x.detach()
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([x, saliencies]):
        if len(img_indices) == 1:
            for column, img in enumerate(target):
                axs[row].imshow(img.permute(1, 2, 0).numpy())
        else:
            for column, img in enumerate(target):
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
    #matplolib 的最后一维是图片的三个通道，但是pytorch中我们的X数据第一维是，所以转换一下维度才能打印正常  permute就是用来做这个的
            #而且转换之前是张量 所以 要转为矩阵
    plt.show()
    plt.close()

def pltxlayer1(layer_index,model,img_indices, train_dataset):

    model.eval()
    model = model.to('cpu')
    model_layers= list(model.children())
    cnnid=0
    filterid=0
    x, y=train_dataset.getbatch(img_indices)
    model.eval()
    hook_handle = model.layer1.register_forward_hook(hook)
    model(x)
    x=x.detach()
    # filter_activations=layer_activations[:, filterid, :, :].detach()
    # hook_handle.remove()

    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    filter_activations = get_k_layer_feature_map(model_layers, layer_index, x)[:,filterid,:,:]
    # filter_activations = []
    # for i,img in enumerate(x):
    #     filter_activations.append(get_k_layer_feature_map(model_layers, layer_index, img)[filterid])
    if len(img_indices) == 1:
        for i, img in enumerate(x):
            axs[0].imshow(img.permute(1, 2, 0))
        for i, img in enumerate(filter_activations):
            axs[1].imshow(normalize(img))
    else:
        for i, img in enumerate(x):
            axs[0][i].imshow(img.permute(1, 2, 0))
        for i, img in enumerate(filter_activations):
            axs[1][i].imshow(normalize(img))
    plt.show()
    plt.close()



def pltxlayer2(model,img_indices, train_dataset):
    plt.rcParams['font.sans-serif']=['STSong']
    #2. 导入数据

    x, y=train_dataset.getbatch(img_indices)
    # 获取第k层的特征图
    '''
    args:
    k:定义提取第几层的feature map
    x:图片的tensor
    model_layer：是一个Sequential()特征层
    '''

    #  可视化特征图
    def show_feature_map(feature_maps, is_save=False, save_path='maps', cmap='gray', map_size:tuple=None, mpa_k:int=-1):
        '''
        :param feature_map: [1, dims, H, W]
        :return: None
        '''
        # 是否对其尺寸
        if map_size:
            feature_map = torch.nn.Upsample(size=map_size, mode='nearest')(feature_map)

        # feature_map = feature_map.squeeze(0)         # [1, 64, 112, 112] -> [64, 112, 112]
        for feature_map in feature_maps:
            feature_map_num = feature_map.shape[0]       #返回通道数
            row_num = np.ceil(np.sqrt(feature_map_num))  # 8   np.ceil  将元素向上取整
            plt.figure()
            for index in range(feature_map_num):         #通过遍历的方式，将64个通道的tensor拿出
                single_dim = feature_map[index] # shape[112, 112]

                plt.subplot(row_num, row_num, index+1) # idx [1...64]
                plt.imshow(single_dim.cpu(), cmap=cmap)
                # plt.imshow(single_dim, cmap='viridis')
                plt.axis('off')

                if is_save:
                    imageio.imwrite( f"./{save_path}/{mpa_k}_" + str(index+1) + ".jpg", single_dim)
            plt.show()

    the_maps_k =7
    # @ 调试这里配合 K
    model_layers= list(model.children())
    # model_layer=model_layers[0] # 这里选择model的第一个模块
    # [1] show single
    feature_maps = get_k_layer_feature_map(model_layers, the_maps_k, x)
    show_feature_map(feature_maps, is_save=False, cmap='hot', map_size=None)


def whereisimportant(model,img_indices, train_dataset):
    def predict(input):
        # input: numpy array, (batches, height, width, channels)

        model.eval()
        input = torch.FloatTensor(input).permute(0, 3, 1, 2)
        # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
        # 也就是 (batches, channels, height, width)

        output = model(input)
        return output.detach().cpu().numpy()

    def segmentation(input):
        # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
        return slic(input, n_segments=100, compactness=1, sigma=1,start_label=1)

    img_indices = [0,1,2,3]
    images, labels = train_dataset.getbatch(img_indices)
    fig, axs = plt.subplots(1, 4, figsize=(15, 8))
    np.random.seed(16)
    # 讓實驗 reproducible
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array

        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
        # classifier_fn 定義圖片如何經過 model_utils 得到 prediction
        # segmentation_fn 定義如何把圖片做 segmentation
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

        lime_img, mask = explaination.get_image_and_mask(
            label=label.item(),
            positive_only=False,
            hide_rest=False,
            num_features=11,
            min_weight=0.05
        )
        # 把 explainer 解釋的結果轉成圖片
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask

        axs[idx].imshow(lime_img)

    plt.show()
    plt.close()


pathClass1 = r'F:\pycharm\deepLeaning\pendi_cv2\data_nor\val'
pathClass2 = r'F:\pycharm\deepLeaning\pendi_cv2\data_abnor\val'
modelpath = 'cat_dog_res18'
model = torch.load(modelpath)
model = model
img_indices=[500,400,250,150]
train_dataset, test_dataset = getDateset(pathClass1, pathClass2,testNum=50)
pltxlayer2(model,img_indices, train_dataset)
pltimportant(model,img_indices,train_dataset)
layerindex = 1
pltxlayer1(layerindex,model,img_indices)
pltxlayer2(model,img_indices)
whereisimportant(model,img_indices,train_dataset=train_dataset)
