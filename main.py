
import os, sys, random

import numpy as np
import scipy.io as sio
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import PVC
from config import save_config
from alignment import alignment
from loss import AverageMeter, PVC_Loss
from utils import euclidean_dist, nan_check, kmeans
from datasets import load_data, Data_Sampler, TrainDataset


def main(config):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.random.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True

    # load dataset
    print("load data ...")
    X_list, Y_list, train_X_list, train_Y_list, test_X_list, test_Y_list = load_data(config)
    n_samples = X_list[0].shape[0]
    print(config['data_name']+', view size:', config['view_size'], ', samples:', n_samples, ', classes:', len(np.unique(Y_list[0])))
    print('training samples', train_X_list[0].shape[0])

    # permutation of second view
    P_index = random.sample(range(n_samples), n_samples)
    P_gt = np.eye(n_samples).astype('float32')
    P_gt = P_gt[:, P_index]

    # data tensor
    var_X_list = []
    var_X_list.append(torch.from_numpy(X_list[0]).to(device))
    var_X_list.append(torch.from_numpy(X_list[1][P_index]).to(device))

    Y_list[1] = Y_list[1][P_index]

    # network
    print ("build model ...")
    # network architecture
    arch_list = []
    for view in range(config['view_size']):
        arch = [X_list[view].shape[1]]
        arch.extend(config['arch'])
        arch_list.append(arch)

    model = PVC(arch_list).to(device)
    criterion = PVC_Loss().to(device)

    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    pretrain(config, model, optimizer_pretrain, train_X_list, criterion, device)

    # P init
    with torch.no_grad():
        ae_encoded, ae_decoded = model(var_X_list)
        C = euclidean_dist(ae_encoded[0], ae_encoded[1])
        P_pred = alignment(C)

    # Training
    optimizer_training = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    P_pred, C = train(config, model, criterion, optimizer_training, var_X_list, P_pred, C, Y_list, device)
    P_pred = P_pred.cpu().detach().numpy()

    # testing
    model.eval()
    ae_encoded, ae_decoded = model(var_X_list)
    
    features1 = ae_encoded[0].cpu().detach().numpy()
    features2 = ae_encoded[1].cpu().detach().numpy()

    features = np.concatenate((features1, np.dot(P_pred, features2)), axis=1)
    y_preds, scores = kmeans(features, Y_list[0])

    return y_preds, scores

def pretrain(config, model, optimizer, X_list, criterion, device):
    print('pretraining ...')
    train_dataset = TrainDataset(X_list)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=config['batch_size'], drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    model.train()
    losses = AverageMeter()

    t_progress = tqdm(range(config['ae_epochs'] + config['pretrain_epoch']), desc='Pretraining')
    for epoch in t_progress:
        current_loss = 0
        count = 0
        for i, (batch_X_list, batch_P) in enumerate(train_loader):
            batch_X_list[0] = torch.squeeze(batch_X_list[0]).to(device)
            batch_X_list[1] = torch.squeeze(batch_X_list[1]).to(device)
            batch_P = torch.squeeze(batch_P).to(device)

            ae_encoded, ae_decoded = model(batch_X_list)

            loss = criterion(batch_X_list, ae_encoded, ae_decoded, batch_P)

            if(epoch>=config['ae_epochs']):
                ce = nn.CrossEntropyLoss()
                C = euclidean_dist(ae_encoded[0], ae_encoded[1])
                P_pred = alignment(C)
                loss += config['lambda'] * F.mse_loss(P_pred, batch_P)

            losses.update(loss.item())
            current_loss+=loss.item()
            count +=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        t_progress.write('epoch %d : loss %.6f'%(epoch, current_loss/count))
        t_progress.set_description_str(' Loss='+str(losses.avg))


def train(config, model, criterion, optimizer, X_list, P_pred, C, Y_list, device):
    print('training ...')
    model.train()
    losses = AverageMeter()
    t_progress = tqdm(range(config['epoch']), desc='Training')
    for epoch in t_progress:
        ae_encoded, ae_decoded = model(X_list)        
        loss = criterion(X_list, ae_encoded, ae_decoded, P_pred)

        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loging
        t_progress.set_description_str(' Loss='+str(loss.item()))
            
    return P_pred, C
