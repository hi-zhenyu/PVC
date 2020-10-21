import torch
import torch.nn as nn 

class PVC_Loss(nn.Module):
    def __init__(self):
        super(PVC_Loss, self).__init__()

    def forward(self, inputs_list, ae_encoded_list, ae_decoded_list, P):
        mse = nn.MSELoss()

        # Loss 1
        Loss1_1 = mse(inputs_list[0], ae_decoded_list[0])
        Loss1_2 = mse(inputs_list[1], ae_decoded_list[1])
        Loss1 = Loss1_1 + Loss1_2

        # Loss 2
        Loss2 = mse(ae_encoded_list[0], torch.mm(P, ae_encoded_list[1]))

        return Loss1 + Loss2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
