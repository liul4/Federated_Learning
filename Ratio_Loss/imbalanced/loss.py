import torch
import torch.nn as nn
import torch.nn.functional as F


class RatioLoss(nn.Module):
    def __init__(self, class_num, alpha=None, size_average=False):
        super(RatioLoss, self).__init__()
        self.alpha = torch.ones(class_num, 1) if alpha is None else alpha
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        # print(inputs)
        # print(P)

        class_mask = torch.zeros(N, C, device=inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        alpha = self.alpha[ids.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = - alpha * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss