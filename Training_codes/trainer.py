#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:48:15 2020

@author: mahyarfazlyab
"""

#from convex_adversarial.dual_network import DualNetwork


import torch

from torch.autograd import Variable

import torch.nn as nn
#import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time
#import gc
# from convex_adversarial import robust_loss, robust_loss_parallel
from convex_adversarial.dual_network import RobustBounds
from convex_adversarial import DualNetwork
import warnings
warnings.filterwarnings('ignore')

import scipy.io
#import numpy as np
#from scipy.io import savemat
#from scipy.linalg import block_diag
from scipy import sparse



def train_baseline(loader, model, opt, epoch, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for t in range(epoch):
        for i, (X,y) in enumerate(loader):
            # X,y = X.cuda(), y.cuda()
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)
            data_time.update(time.time() - end)

            # out = model(Variable(X))
            out = model(X)
            # ce = nn.CrossEntropyLoss()(out, Variable(y))
            ce = nn.MSELoss()(out, Variable(y))

            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
            
            opt.zero_grad()
            ce.backward()
            opt.step()

            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(ce.item(), X.size(0))
            errors.update(err, X.size(0))

        print('epoch: ',t,'CrossEntropyLoss: ',ce.item())



        #print(epoch, i, ce.item(), err)
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
               epoch, i, len(loader), batch_time=batch_time,
               data_time=data_time, loss=losses, errors=errors))
        #log.flush()


    # DEBUG = False

def train_lip_bound(loader, model, lam, opt, epoch, verbose):

    '''
    Train a neural net by constraining the lipschitz constant of each layer
    '''

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for t in range(epoch):
        for i, (X,y) in enumerate(loader):
            #X,y = X.cuda(), y.cuda()
            batch_size = X.shape[0]
            X = X.view(batch_size, -1)
            data_time.update(time.time() - end)

            out = model(Variable(X))
            ce = nn.MSELoss()(out, Variable(y))
            # ce=nn.GaussianNLLLoss()(out, Variable(y))
            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

            opt.zero_grad()
            ce.backward()
            opt.step()

            num_layers = int((len(model)-1)/2)
            for c in range(num_layers+1):
                scale = max(1,np.linalg.norm(model[2*c].weight.data,2)/lam)
                model[2*c].weight.data = model[2*c].weight.data/scale


            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(ce.item(), X.size(0))
            errors.update(err, X.size(0))

            if verbose and i % verbose == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                    t, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, errors=errors))

            # if i == 2000:
            #     break

        print('epoch: ',t,'CrossEntropyLoss: ',ce.item())



def train_robust(loader, model, opt, epsilon, epoch, verbose, real_time=False, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.train()

    end = time.time()
    for t in range(epoch):
        for i, (X,y) in enumerate(loader):
            #X,y = X.cuda(), y.cuda().long()
            if y.dim() == 2:
                y = y.squeeze(1)
            data_time.update(time.time() - end)

            with torch.no_grad():
                #out = model(Variable(X))
                batch_size = X.shape[0]
                out = model(Variable(X.view(batch_size, -1)))
                ce = nn.MSELoss()(out, Variable(y))
                err = (out.max(1)[1] != y).float().sum()  / X.size(0)


            robust_ce, robust_err = robust_loss(model, epsilon, Variable(X.view(batch_size, -1)), Variable(y))
            opt.zero_grad()
            robust_ce.backward()


            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            # measure accuracy and record loss
            losses.update(ce.item(), X.size(0))
            errors.update(err.item(), X.size(0))
            robust_losses.update(robust_ce.detach().item(), X.size(0))
            robust_errors.update(robust_err, X.size(0))

            # measure elapsed time
            batch_time.update(time.time()-end)
            end = time.time()

            #print(epoch, i, robust_ce.detach().item(), robust_err, ce.item(), err.item())

            if verbose and i % verbose == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                      'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                       epoch, i, len(loader), loss=losses, errors=errors,
                       rloss = robust_losses, rerrors = robust_errors), end=endline)

        print('epoch: ',t,'CrossEntropyLoss: ',robust_ce.item())
    #torch.cuda.empty_cache()


def train_madry(loader, model, epsilon, opt, epoch, verbose):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    plosses = AverageMeter()
    perrors = AverageMeter()

    model.train()

    end = time.time()
    for t in range(epoch):
        for i, (X,y) in enumerate(loader):
            data_time.update(time.time() - end)

            batch_size = X.shape[0]
            X = X.view(batch_size, -1)

            # # perturb
            X_pgd = Variable(X, requires_grad=True)
            #X_pgd = Variable(X.view(BATCH_SIZE, -1),requires_grad=True)
            #print(X.view(BATCH_SIZE, -1).shape)
            for _ in range(40):
                opt_pgd = optim.Adam([X_pgd], lr=1e-3)
                opt.zero_grad()
                # loss = nn.CrossEntropyLoss()(model(X_pgd), Variable(y))
                loss = nn.MSELoss()(model(X_pgd), Variable(y))
                loss.backward()
                eta = 0.01*X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

                # adjust to be within [-epsilon, epsilon]
                eta = torch.clamp(X_pgd.data - X, -epsilon, epsilon)
                X_pgd.data = X + eta
                X_pgd.data = torch.clamp(X_pgd.data, 0, 1)

            out = model(Variable(X))
            # ce = nn.CrossEntropyLoss()(out, Variable(y))
            ce = nn.MSELoss()(out, Variable(y))
            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

            pout = model(Variable(X_pgd.data))
            pce = nn.MSELoss()(pout, Variable(y))
            perr = (pout.data.max(1)[1] != y).float().sum()  / X.size(0)

            opt.zero_grad()
            pce.backward()
            opt.step()

            batch_time.update(time.time()-end)
            end = time.time()
            losses.update(ce.item(), X.size(0))
            errors.update(err, X.size(0))
            plosses.update(pce.item(), X.size(0))
            perrors.update(perr, X.size(0))

        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'PGD Loss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                  'PGD Error {perrors.val:.3f} ({perrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, errors=errors,
                      ploss=plosses, perrors=perrors))
        print('epoch: ',t,'CrossEntropyLoss: ',ce.item())

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

def robust_loss(net, epsilon, X, y,
                size_average=True, device_ids=None, parallel=False, **kwargs):
    if parallel:
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else:
        f = RobustBounds(net, epsilon, **kwargs)(X,y)
    err = (f.max(1)[1] != y)
    if size_average:
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err

def evaluate_baseline(loader, model):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X,y) in enumerate(loader):
        #X,y = X.cuda(), y.cuda()
        #out = model(Variable(X))
        TEST_SIZE = X.shape[0]
        #out = model(Variable(X.view(TEST_SIZE, -1)))
        out = model(X.view(TEST_SIZE, -1))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
    return errors.avg