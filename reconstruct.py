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
    model_pre = models_mae.__dict__[args.pre_model](norm_pix_loss=False, guide_mask=True)
    model_preori = models_mae.__dict__[args.pre_model](norm_pix_loss=False)
    model_path = f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}_guide.bin'
    # model_path =  f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}_wopre.bin'
    # model_path = f'{args.groundPredsave}/{int(mask_ratio * 100)}/model_epoch_{pre_epo}_wocut.bin'
    checkpoint = torch.load(model_path, map_location='cpu')
    checkpoint_model = checkpoint['model_state_dict']
    mas_pre = model_pre.load_state_dict(checkpoint_model, strict=True)
    msg_vit = model_vit.load_state_dict(checkpoint_model , strict=False)

    checkpoint_ori = torch.load(args.predPath_ori, map_location='cpu')['model']
    msg_pre_ori = model_preori.load_state_dict(checkpoint_ori, strict=True)
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

learning_rate = 1e-5
w = 0.00001
criterion =nn.CrossEntropyLoss()

pre_epo = 20
epoch = 20
# w = 0.00001
savePath = 'model_save/max_auc/WOGP_75'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    seed_everything(1022)
    # random.seed(0)
    # mask_ratios = [0.75]
    mask_ratios = [0.5,0.75]
    # mask_ratios = [0.65, 0.70, 0.75, 0.80, 0.85]
    args = get_args_parser()
    args = args.parse_args()

    model_pre = models_mae.__dict__[args.pre_model](norm_pix_loss=False, guide_mask=True)
    imgpath = '/home/dataset/pendi/abnormal/V1354_1.jpg'
    # imgpath = r'testPic/4.jpg'

    all_acc = []

    for i, mask_ratio in enumerate(mask_ratios):
        print(mask_ratio)
        model_vit, model_vit_ori, model_mae, model_mae_ori, model_vit_wo_train = initMaeClass(args, mask_ratio)

        run_one_image(imgpath, model_mae,mask_ratio)
    #     # plt_image(imgpath,model_mae,mask_ratio=mask_ratio)
    #     # run_one_image(imgpath, model_mae, mask_ratio)
    #
        setup_seed(0)
        # mask_ratio = 0.75
        run_one_image(imgpath, model_mae_ori,mask_ratio)

        model = models_mae.__dict__[args.pre_model](norm_pix_loss=False, guide_mask=True)
        savePath = 'model_save/groundpre/' + str(int(mask_ratio*100))+'/model_epoch_20.bin'
        checkpoint = torch.load(args.predPath_ori, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=False)

        # model = torch.load(savePath)
        # model.eval()
        run_one_image(imgpath, model,mask_ratio)
