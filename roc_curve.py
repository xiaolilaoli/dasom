import os
import numpy as np
import torch
import torch.nn as nn

import random
import models_mae
import argparse
import torch
import timm
assert timm.__version__ == "0.5.4" # version check
import models_vit
from torch import optim
from model_utils.train import train_VAL
from model_utils.data import getDateset
from model_utils.model import initialize_model
from transformers import AdamW, get_linear_schedule_with_warmup
from util.misc import save_model,load_model
from tqdm import tqdm
from main import run_one_image, plt_image

from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import roc_curve, auc,confusion_matrix, f1_score, precision_recall_curve, average_precision_score,precision_score,roc_auc_score
from itertools import cycle







def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    #model
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classfication types')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device')

    parser.add_argument('--drop_path', default=0.1, type=float, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pre_model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    #path
    parser.add_argument('--predModelPath', default='/home/reapper/mae/model_save/groundpre/model_epoch_2000_loss_0.10522957146167755.bin',
                        help='finetune from checkpoint')
    parser.add_argument('--groundPredPath', default='model_save/ground_pre/mae_ground_pre',
                        help='pre_save')
    parser.add_argument('--predPath_ori', default='model_save/mae_visualize_vit_base.pth',
                        help='pre_model_ori_mae')
    parser.add_argument('--groundPredsave', default='model_save/groundpre',
                        help='finetune from checkpoint')
    parser.add_argument('--dataPath', default='/home/dataset/pendi/',
                        help='dataPath')

    # parser.add_argument('--dataPath', default='E:\dataset\pendi',
    #                     help='dataPath')
    ################data#######################
    parser.add_argument('--testNum', default=50,
                        help='testNum')
    parser.add_argument('--testpos', default='rear',
                        help='testpos')
    parser.add_argument('--input_size', default=224,
                        help='input_size')
    parser.add_argument('--batch_size', default=64,
                        help='batch_size')
    ##################train##########################
    parser.add_argument('--lr', default=1e-5,
                        help='batch_size')
    return parser


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    large_lr = ['']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer_grouped_parameters = [
    #     {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and not any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
    #      'lr': args.learning_rate, 'weight_decay': 0.0},
    #     {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
    #      'lr': args.bert_learning_rate, 'weight_decay': args.weight_decay},
    #     {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
    #      'lr': args.bert_learning_rate, 'weight_decay': 0.0},
    # ]
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler

def roc_train(model,train_set,val_set,optimizer,scheduler,loss,batch_size,w,num_epoch,device, save_, acc2=None, roc_value=None):
    # train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    # val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)
    train_loader= train_set
    val_loader = val_set
    # 用测试集训练模型model(),用验证集作为测试集来验证
    plt_train_loss = [0]
    plt_val_loss = []
    plt_train_acc = [0]
    plt_val_acc = []
    maxacc = 0


    # update_lr(optimizer,epoch)
    epoch_start_time = time.time()
    val_acc = 0.0
    val_loss = 0.0
    #验证集val
    model.eval()
    labels = []
    preds = []
    score_list = []
    max_roc = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            if type(val_pred) != torch.Tensor:
                val_pred = val_pred.logits
            # batch_loss = loss(val_pred, data[1].cuda(),w, model)
            batch_loss = loss(val_pred, data[1].to(device))
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
            labels.extend(data[1].numpy())
            preds.extend(np.argmax(val_pred.cpu().data.numpy(), axis=1))
            score_tmp = nn.Softmax(dim=1)(val_pred)
            score_list.extend(score_tmp.detach().cpu().numpy())


######################计算TN,FN等。
        preds_tensor = torch.tensor(preds)
        label_tensor = torch.tensor(labels)
        # TP predict 和 label 同时为1

###############################################
        #########计算其他指标
        auc_score = roc_auc_score(labels, np.array(score_list)[:,1])
        conf_mat = confusion_matrix(labels,preds)
        TP = conf_mat.diagonal()
        FP = conf_mat.sum(1) - TP
        FN = conf_mat.sum(0) - TP
        # TP = TP.float()  # convert to float dtype for the next calculation
        # FP = FP.float()
        # FN = FN.float()

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        m_precision = precision.mean()
        m_recall = recall.mean()
        m_F1 = 2*(m_recall*m_precision)/(m_recall+m_precision)

        if auc_score> max_roc:
            max_roc = auc_score
            fpr, tpr, thresholds = roc_curve(labels, np.array(score_list)[:,1])

        plt_val_acc.append(val_acc/val_set.dataset.__len__())
        plt_val_loss.append(val_loss/val_set.dataset.__len__())

        #将结果 print 出來
        print('[%03d/%03d] %2.2f sec(s) | Train Acc: %3.6f loss: %3.6f Val Acc: %3.6f loss: %3.6f auc_score: %3.6f m_F1: %3.6f m_rec:%3.6f' % \
              (1, 1, time.time()-epoch_start_time, \
                plt_train_acc[-1], plt_train_loss[-1],plt_val_acc[-1], plt_val_loss[-1],auc_score,m_F1,m_recall))


    shape = ['bv-', 'kp-', 'gs-', 'rh-']
    # Accuracy曲线

    # my_x_ticks = np.arange(0, 20, 4)
    # Accuracy曲线
    plt.plot(fpr, tpr , "c*-")
    # plt.xticks(my_x_ticks)
    # plt.ylim((0, 1))
    # plt.xlabel('epoch')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if roc_value != None:
        for i,each in enumerate(roc_value):
            plt.plot(each[0], each[1], shape[i])
        plt.title('ROC')
        # plt.legend(['ViT Base', 'ViT MaePre', 'ViT GroundPre_0.5', 'ViT GroundPre_0.75', 'ViT GroundPre_0.9'])
        plt.legend(['GSPPD ViT-B', 'GSPPD wo G', 'GSPPD_0.5', 'GSPPD_0.75', 'GSPPD_0.9'])
        plt.savefig('pic/few_shot', dpi=600, bbox_inches='tight')
        plt.show()
    else:
        plt.title('ROC')
        plt.legend(['GSPPD_0.75'])
        # plt.savefig('acc.png')
        plt.show()

    return [fpr, tpr]



def initMaeClass(args,mask_ratio=0.75):
    model_vit = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    model_vit_ori = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    model_vit_wo_train = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    model_pre = models_mae.__dict__[args.pre_model](norm_pix_loss=False)
    model_preori = models_mae.__dict__[args.pre_model](norm_pix_loss=False)
    model_path = f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}.bin'
    # model_path =  f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}_wopre.bin'
    # model_path = f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}_wocut.bin'
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model = checkpoint['model_state_dict']
    mas_pre = model_pre.load_state_dict(checkpoint_model, strict=True)
    msg_vit = model_vit.load_state_dict(checkpoint_model , strict=False)

    checkpoint_ori = torch.load(args.predPath_ori, map_location='cpu')['model']
    # msg_pre_ori = model_preori.load_state_dict(checkpoint_ori, strict=True)
    msg_vit_ori = model_vit_ori.load_state_dict(checkpoint_ori, strict=False)
    # print(msg_vit)
    return model_vit, model_vit_ori, model_pre, model_preori, model_vit_wo_train


##################################################################




def ground_pre(model,mae_loader,mask_ratio=0.75):

    checkpoint = torch.load(args.predPath_ori, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    msg = model.load_state_dict(checkpoint_model, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.95))
    print(optimizer)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000,
    #                                             num_training_steps=100*len(mae_loader.dataset)/args.batch_size)
    scheduler = None
    model = torch.nn.parallel.DataParallel(model.to(args.device))
    model.train()
    save_path = f'{args.groundPredsave}/{int(mask_ratio*100)}/'
    os.makedirs(save_path, exist_ok=True)
    for epo in tqdm(range(pre_epo+1)):
        for data,label in mae_loader:
            loss, _, _ = model(data, mask_ratio=mask_ratio)
            loss_value = loss.mean()
            loss_value.backward()
            optimizer.step()
            model.zero_grad()
            # scheduler.step()
        print(loss)

        if epo == 20:
            torch.save({'epoch': epo, 'model_state_dict': model.module.state_dict(), 'loss': loss},
                       save_path+f'model_epoch_{epo}.bin')

#################################################################

learning_rate = 1e-4
w = 0.00001
criterion =nn.CrossEntropyLoss()

pre_epo = 20
epoch = 20
# w = 0.00001


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
##################################################################


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)




if __name__ == '__main__':
    seed_everything(0)
    # random.seed(0)
    # mask_ratios = [0.75]
    mask_ratios = [0.4, 0.6, 0.75, 0.80, 0.9]
    # mask_ratios = [0.65, 0.70, 0.75, 0.80, 0.85]
    args = get_args_parser()
    args = args.parse_args()
    imgpath = r'testPic/6.jpg'
    model_pre = models_mae.__dict__[args.pre_model](norm_pix_loss=False)
    # train_loader, val_loader, mae_loader = getDateloader(args)

    savePath_ = 'model_save/max_auc/'
    class0 = r'/home/dataset/pendi/abnormal'     #类别0
    class1 = r'/home/dataset/pendi/normal'
    train_dataset, test_dataset,mae_dataset = getDateset(class0, class1, testNum=100)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
    mae_loader = DataLoader(mae_dataset, batch_size=args.batch_size, shuffle=True)
    # model_name = 'resnet18'
    # savePath = f'model_save/{model_name}_line'
    #
    # train_dataset, val_dataset = getDateset(class0, class1, testNum=100)
    # model, input_size = initialize_model(model_name, 2, False, True)
    all_acc = []
    roc_value = []
    # setup_seed(1)
    model_path_lists = [
        '/home/reapper/mae/model_save/max_auc/ViTmax',
        '/home/reapper/mae/model_save/max_auc/WOGP_75max',
        '/home/reapper/mae/model_save/max_auc/0.4max',
        '/home/reapper/mae/model_save/max_auc/0.75max',
        '/home/reapper/mae/model_save/max_auc/0.9max',
    ]
    model_path_lists = [
        '/home/reapper/mae/model_save/max_auc/FEW_ViTmax',
        '/home/reapper/mae/model_save/max_auc/WOGP_75max',
        '/home/reapper/mae/model_save/max_auc/0.4max',
        '/home/reapper/mae/model_save/max_auc/0.75max',
        '/home/reapper/mae/model_save/max_auc/0.9max',
    ]
    model_path_lists = [
        '/home/reapper/mae/model_save/max_auc/0.1guidemax',
        '/home/reapper/mae/model_save/max_auc/0.4guidemax',
        '/home/reapper/mae/model_save/max_auc/0.5guidemax',
        '/home/reapper/mae/model_save/max_auc/0.6guidemax',
        '/home/reapper/mae/model_save/max_auc/0.7guidemax',
        '/home/reapper/mae/model_save/max_auc/0.8guidemax',
        '/home/reapper/mae/model_save/max_auc/0.9guidemax',
    ]


    for model_path in model_path_lists:
        model = torch.load(model_path).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print(optimizer)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(len(train_dataset)// args.batch_size),
        #                                             num_training_steps=30 * len(train_dataset) // args.batch_size)
        # model = torch.nn.parallel.DataParallel(model.cuda())
        scheduler = None
        savePath = None
        fp_tp = roc_train(model,train_loader, val_loader, optimizer,scheduler, criterion, args.batch_size, w, num_epoch=epoch,save_=savePath,device=device)
        #

        roc_value.append(fp_tp)


    shape = ["c*-", 'bv-', 'kp-', 'gs-', 'rh-']

    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for i,each in enumerate(roc_value):
        plt.plot(each[0], each[1], shape[i])
    plt.title('ROC')
    # plt.legend(['ViT Base', 'ViT MaePre', 'ViT GroundPre_0.5', 'ViT GroundPre_0.75', 'ViT GroundPre_0.9'])
    plt.legend(['GSPPD ViT-B', 'GSPPD wo G', 'GSPPD_0.4', 'GSPPD_0.75', 'GSPPD_0.9'])
    plt.savefig('pic/roc_curve', dpi=600, bbox_inches='tight')
    plt.show()