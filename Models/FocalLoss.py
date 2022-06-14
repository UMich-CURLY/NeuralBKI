# Focal Loss
# https://arxiv.org/pdf/1708.02002v2.pdf]
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, device="cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, device=device)
        self.alpha = alpha
        self.device = device

        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha], device=device)

    def forward(self, input, target):
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, 1, target).reshape(-1)
        pt = Variable(torch.exp(logpt.data))
        
        alpha_weight = 1
       
        if self.alpha is not None:
            alphat = torch.gather(self.alpha, 0, target.reshape(-1))
            # alphat = torch.gather(
            #     self.alpha, 0, target.data.reshape(-1))
            # logpt = logpt * Variable(alphat)
            alpha_weight = alphat

        loss = -alpha_weight * torch.pow(1-pt, self.gamma) * logpt
        # pdb.set_trace()
        return torch.mean(loss)