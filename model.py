import math

import torch
import torch.nn as nn 


class PVC(nn.Module):
    def __init__(self, arch_list):
        super(PVC, self).__init__()
        self.view_size = len(arch_list)
        self.enc_list = nn.ModuleList()
        self.dec_list = nn.ModuleList()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        # network
        for view in range(self.view_size):
            enc, dec = self.single_ae(arch_list[view])
            self.enc_list.append(enc)
            self.dec_list.append(dec)
        self.dim = arch_list[0][0]
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.dim)

    def single_ae(self, arch):
        # encoder
        enc = nn.ModuleList()
        for i in range(len(arch)):
            if i < len(arch)-1:
                enc.append(nn.Linear(arch[i], arch[i+1]))
            else:
                break
        
        # decoder
        arch.reverse()
        dec = nn.ModuleList()
        for i in range(len(arch)):
            if i < len(arch)-1:
                dec.append(nn.Linear(arch[i], arch[i+1]))
            else:
                break
        
        return enc, dec

    def forward(self, inputs_list):
        encoded_list = []
        decoded_list = []

        for view in range(self.view_size):
            # encoded
            encoded = inputs_list[view]
            for i, layer in enumerate(self.enc_list[view]):
                if i < len(self.enc_list[view]) - 1:
                    encoded = self.relu(layer(encoded))
                else: # the last layer
                    encoded = layer(encoded)
            encoded_list.append(encoded)

            # decoded
            decoded = encoded
            for i, layer in enumerate(self.dec_list[view]):
                if i < len(self.dec_list[view]) - 1:
                    decoded = self.relu(layer(decoded))
                else: # the last layer
                    decoded = layer(decoded)
            decoded_list.append(decoded)

        return encoded_list, decoded_list
