import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import roc_curve, auc,confusion_matrix, f1_score, precision_recall_curve, average_precision_score,precision_score,roc_auc_score
from itertools import cycle
from scipy.interpolate import interp1d,interp2d,interpolate
import scipy

#更新学习率
def update_lr(optimizer,epoch):
    if epoch<45:
        lr = 1e-3
    elif epoch<75:
        lr = 0.0005
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            # if m.bias:
            #     init.constant(m.bias, 0)

def Heinit(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                          nonlinearity='relu')    # 传说中的凯明初始化
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)

def train_VAL(model,train_set,val_set,optimizer,scheduler,loss,batch_size,w,num_epoch,device, save_, acc2=None, roc_value=None):
    # train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    # val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=0)
    train_loader= train_set
    val_loader = val_set
    # 用测试集训练模型model(),用验证集作为测试集来验证
    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []
    maxacc = 0

    for epoch in range(num_epoch):
        # update_lr(optimizer,epoch)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train() # 确保 model_utils 是在 训练 model_utils (开启 Dropout 等...)

        time_sum = 0
        time_num = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad() # 用 optimizer 将模型参数的梯度 gradient 归零

            pred_time_s = time.time()
            train_pred = model(data[0].to(device)) # 利用 model_utils 得到预测的概率分布，这边实际上是调用模型的 forward 函数
            pred_time_e = time.time()
            # print("%2.6f sec(s)", pred_time_e-pred_time_s)
            time_sum += pred_time_e-pred_time_s
            time_num +=1
            # batch_loss = loss(train_pred, data[1].cuda(), w, model) # 计算 loss （注意 prediction 跟 label 必须同时在 CPU 或是 GPU 上）
            if type(train_pred)!= torch.Tensor:
                train_pred = train_pred.logits
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward() # 利用 back propagation 算出每个参数的 gradient
            optimizer.step() # 以 optimizer 用 gradient 更新参数
            if scheduler != None:
                scheduler.step()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        print("%2.6f sec(s)"%(time_sum/time_num) )

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

            TP = TN = FN = FP = 0

############计算 auc等指标
            # score_array = np.array(score_list)
            # num_class =2
            #
            # label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
            # label_onehot = torch.zeros(label_tensor.shape[0], num_class)
            # label_onehot.scatter_(dim=1, index=label_tensor, value=1)
            # label_onehot = np.array(label_onehot)
            #
            # fpr_dict = dict()
            # tpr_dict = dict()
            # roc_auc_dict = dict()
            # for i in range(num_class):
            #     fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            #     roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            #
            # # micro
            # fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
            # roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
            #
            # # macro
            # # First aggregate all false positive rates
            # all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
            # # Then interpolate all ROC curves at this points
            # mean_tpr = np.zeros_like(all_fpr)
            # for i in range(num_class):
            #     mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
            # # Finally average it and compute AUC
            # mean_tpr /= num_class
            # fpr_dict["macro"] = all_fpr
            # tpr_dict["macro"] = mean_tpr
            # roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
            #
            # plt.figure()
            # lw = 2
            # plt.plot(fpr_dict["micro"], tpr_dict["micro"],
            #          label='micro-average ROC curve (area = {0:0.2f})'
            #                ''.format(roc_auc_dict["micro"]),
            #          color='deeppink', linestyle=':', linewidth=4)
            #
            # plt.plot(fpr_dict["macro"], tpr_dict["macro"],
            #          label='macro-average ROC curve (area = {0:0.2f})'
            #                ''.format(roc_auc_dict["macro"]),
            #          color='navy', linestyle=':', linewidth=4)
            #
            # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            # for i, color in zip(range(num_class), colors):
            #     plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
            #              label='ROC curve of class {0} (area = {1:0.2f})'
            #                    ''.format(i, roc_auc_dict[i]))
            # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Some extension of Receiver operating characteristic to multi-class')
            # plt.legend(loc="lower right")
            # plt.savefig('set113_roc.jpg')
            # plt.show()
            #

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


            if val_acc > maxacc:
                torch.save(model,save_+'max')
                maxacc = val_acc

            plt_train_acc.append(train_acc/train_set.dataset.__len__())
            plt_train_loss.append(train_loss/train_set.dataset.__len__())
            plt_val_acc.append(val_acc/val_set.dataset.__len__())
            plt_val_loss.append(val_loss/val_set.dataset.__len__())

            #将结果 print 出來
            print('[%03d/%03d] %2.2f sec(s) | Train Acc: %3.6f loss: %3.6f Val Acc: %3.6f loss: %3.6f auc_score: %3.6f m_F1: %3.6f m_rec:%3.6f' % \
                  (epoch + 1, num_epoch, time.time()-epoch_start_time, \
                    plt_train_acc[-1], plt_train_loss[-1],plt_val_acc[-1], plt_val_loss[-1],auc_score,m_F1,m_recall))

        if epoch == num_epoch-1:
            torch.save(model,save_ + 'final')

    # # Loss曲线
    # plt.plot(plt_train_loss)
    # plt.plot(plt_val_loss)
    # plt.title('Loss')
    # plt.legend(['train', 'val'])
    # plt.savefig('loss.png')
    # plt.show()
    #
    # # Accuracy曲线
    # plt.plot(plt_train_acc)
    # plt.plot(plt_val_acc)
    # plt.title('Accuracy')
    # plt.legend(['train', 'val'])
    # plt.savefig('acc.png')
    # plt.show()
    shape = ['bv-', 'kp-', 'gs-', 'rh-']
    my_x_ticks = np.arange(0, 20, 4)
    # Accuracy曲线
    plt.plot(plt_val_acc, "c*-")
    plt.xticks(my_x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('epoch')
    if acc2 !=None:
        for i,each in enumerate(acc2):
            plt.plot(each, shape[i])
        plt.title('Accuracy')
        # plt.legend(['ViT Base', 'ViT MaePre', 'ViT GroundPre_0.5', 'ViT GroundPre_0.75', 'ViT GroundPre_0.9'])
        # plt.legend([' ViT-B', 'GGMIM wo GT', 'GGMIM wo GU_0.5', 'GGMIM wo GU_0.75', 'GGMIM wo GU_0.9'])
        plt.legend(['GGMIM wo GT', ' ViT-B', 'GGMIM_0.5', 'GGMIM_0.75', 'GGMIM_0.9'])
        plt.savefig('pic/few_shot', dpi=600, bbox_inches='tight')
        plt.show()
    else:
        plt.title('Accuracy')
        plt.legend(['wo_ground_pre'])
        plt.savefig('acc.png')
        plt.show()

    #     # my_x_ticks = np.arange(0, 20, 4)
    #     # Accuracy曲线
    #     plt.plot(fpr, tpr , "c*-")
    #     # plt.xticks(my_x_ticks)
    #     # plt.ylim((0, 1))
    #     # plt.xlabel('epoch')
    #     plt.xlabel('FPR')
    #     plt.ylabel('TPR')
    # if roc_value != None:
    #     for i,each in enumerate(roc_value):
    #         plt.plot(each[0], each[1], shape[i])
    #     plt.title('ROC')
    #     # plt.legend(['ViT Base', 'ViT MaePre', 'ViT GroundPre_0.5', 'ViT GroundPre_0.75', 'ViT GroundPre_0.9'])
    #     plt.legend([' ViT-B', 'GGMIM wo GT', 'GGMIM wo GU_0.5', 'GGMIM wo GU_0.75', 'GGMIM wo GU_0.9'])
    #     plt.savefig('pic/few_shot_ROC', dpi=600, bbox_inches='tight')
    #     plt.show()
    # else:
    #     plt.title('ROC')
    #     plt.legend(['GSPPD_0.75'])
    #     # plt.savefig('acc.png')
    #     plt.show()






    return plt_val_acc , [fpr, tpr]
