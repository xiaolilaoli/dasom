import torch.nn as nn
import torch
import numpy as np
def getLoss(pred, target, w, model):
    weights = torch.from_numpy(np.array([0.05,1])).float().cuda()
    loss = nn.CrossEntropyLoss()
    ''' Calculate loss '''
    regularization_loss = 0
    for param in model.parameters():
        # TODO: you may implement L1/L2 regularization here
        # 使用L2正则项
        # regularization_loss += torch.sum(abs(param))
        regularization_loss += torch.sum(param ** 2)
    return loss(pred, target) + w * regularization_loss

loss =  getLoss