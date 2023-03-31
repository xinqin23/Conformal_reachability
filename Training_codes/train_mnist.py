
from trainer import *
from functions import export2matlab
# from convex_adversarial import *
# from convex_adversarial.dual_network import *

# import sys
# sys.path.append("convex_adversarial/")
# from convex_adversarial import robust_loss


import torch

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time
import gc
import warnings
warnings.filterwarnings('ignore')

import scipy.io
import numpy as np
from scipy.io import savemat
from scipy.linalg import block_diag
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
# import hdf5storage


def main():

    print(2020)
    train_batch_size = 2

    transform = transforms.ToTensor()
    
    
    data = loadmat('Data')
    # data = hdf5storage.loadmat('DPL_Data_gym.mat')
    # data = loadmat('s2s_Data_trajectory_exact.mat')


    Xtrain = data['Input']
    # Xtrain = torch.Tensor(np.transpose(Xtrain[:, :1999998]))
    Xtrain = torch.Tensor(np.transpose(Xtrain[:, :1400000]))

    # Xtrain = Xtrain.cuda()
    Ytrain = data['Output']
    Ytrain = torch.Tensor(np.transpose(Ytrain[:, :1400000]))
    # Ytrain = Ytrain.cuda()
    print(Xtrain.shape)
    print(Ytrain.shape)
    
    
    trainset=TensorDataset(Xtrain, Ytrain)
    # trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=6)
    
    print('Train data loaded')
    # testset=TensorDataset(testX, textlabelX)
    # # testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
    # test_size = testset.__len__()
    # testloader = torch.utils.data.DataLoader(testset, batch_size=test_size, shuffle=False, num_workers=2)

    # print('Test data loaded')


    net = nn.Sequential(
        nn.Linear(2,20),
        nn.ReLU(),
        nn.Linear(20,20),
        nn.ReLU(),
        nn.Linear(20,20),
        nn.ReLU(),
        nn.Linear(20,2)
    )

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    epsilon = 0.02
    verbose = True
    epoch = 8
    print('Training started')

    # train_baseline(trainloader, net, optimizer, epoch, verbose)
    # train_madry(trainloader, net, epsilon, optimizer, epoch, verbose)
    # train_robust(trainloader, net, optimizer, epsilon, epoch, verbose, real_time=False, clip_grad=None)
    train_lip_bound(trainloader, net, 3 , optimizer, epoch, verbose)  # lip for bench mark 1 is 1.0270

    # print('Evaluating accuracy on test set')

    # err = evaluate_baseline(testloader, net)
    # print('Accuracy: ', 1-err.item())

    export2matlab('s2s_Model',net)


if __name__ == '__main__':
    main()
