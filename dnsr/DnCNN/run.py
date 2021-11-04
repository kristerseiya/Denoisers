
import torch
import torch.nn.functional as F
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import sys

from . import data
from . import model

def get_gauss2d(h, w, sigma):
    gauss_1d_w = np.array([np.exp(-(x-w//2)**2/float(2**sigma**2)) for x in range(w)])
    gauss_1d_w = gauss_1d_w / gauss_1d_w.sum()
    gauss_1d_h = np.array([np.exp(-(x-h//2)**2/float(2**sigma**2)) for x in range(h)])
    gauss_1d_h = gauss_1d_h
    gauss_2d = np.array([gauss_1d_w * s for s in gauss_1d_h])
    gauss_2d = gauss_2d / gauss_2d.sum()
    return gauss_2d

def train_single_epoch(net, optimizer, train_loader,
                       noise_lvl, clip=False, lossfn='L2',
                       scheduler=None):

    n_data = 0
    sigma = noise_lvl

    if lossfn.upper() == 'L2':
        lossfn = F.mse_loss
    elif lossfn.upper() == 'L1':
        lossfn = F.l1_loss

    total_loss = 0.
    net.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=False, file=sys.stdout)

    filter = get_gauss2d(5, 5, 1)
    filter = torch.from_numpy(filter)
    filter = filter.unsqueeze(0)

    for images in train_loader:
        optimizer.zero_grad()
        batch_size = images.size(0)
        images = images.to(net.device)
        if type(noise_lvl) == list:
            # sigma = torch.rand(batch_size, 1, 1, 1, device=net.device)
            sigma = torch.rand_like(images)
            sigma = sigma * (noise_lvl[1] - noise_lvl[0]) + noise_lvl[0]
            sigma = F.conv2d(sigma, filter, padding='same')
            # sigma = torch.sqrt((noise_lvl[1] - noise_lvl[0])**2 * sigma) + noise_lvl[0]
        noise = torch.randn_like(images) * sigma / 255.
        noisy = images + noise
        if clip:
            noisy = torch.clip(noisy, 0, 1)
        if isinstance(net, model.cDnCNN):
            # condition = (sigma / 255.).expand_as(noisy)
            condition = sigma / 255.
            output = net(noisy, condition)
        else:
            output = net(noisy)
        loss = lossfn(output, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        n_data += batch_size
        if scheduler != None:
            scheduler.step()
        pbar.update(1)

    tqdm.close(pbar)

    return total_loss / float(n_data)

@torch.no_grad()
def validate(net, test_loader, noise_lvl, clip=False, lossfn='L2'):

    n_data = 0
    sigma = noise_lvl

    if lossfn.upper() == 'L2':
        lossfn = F.mse_loss
    elif lossfn.upper() == 'L1':
        lossfn = F.l1_loss

    total_loss = 0.
    net.eval()

    pbar = tqdm(total=len(test_loader), position=0, leave=False, file=sys.stdout)

    filter = get_gauss2d(5, 5, 1)
    filter = torch.from_numpy(filter)
    filter = filter.unsqueeze(0)
    
    for images in test_loader:
        batch_size = images.size(0)
        images = images.to(net.device)
        if type(noise_lvl) == list:
            # sigma = torch.rand(batch_size, 1, 1, 1, device=net.device)
            sigma = torch.rand_like(images)
            sigma = sigma * (noise_lvl[1] - noise_lvl[0]) + noise_lvl[0]
            sigma = F.conv2d(sigma, filter, padding='same')
            # sigma = torch.sqrt((noise_lvl[1] - noise_lvl[0])**2 * sigma) + noise_lvl[0]
        noise = torch.randn_like(images) * sigma / 255.
        noisy = images + noise
        if clip:
            noisy = torch.clip(noisy, 0, 1)
        if isinstance(net, model.cDnCNN):
            # condition = (sigma / 255.).expand_as(noisy)
            condition = sigma / 255.
            output = net(noisy, condition)
        else:
            output = net(noisy)
        total_loss += lossfn(output, images).item() * batch_size
        n_data += batch_size
        pbar.update(1)

    tqdm.close(pbar)

    return total_loss / float(n_data)


def train(net, optimizer, max_epoch, train_loader, noise_lvl, clip=False, lossfn='L2',
          validation=None, scheduler=None, lr_step='epoch',
          checkpoint_dir=None, max_tolerance=-1):

    best_loss = 99999.
    tolerated = 0

    if lr_step == 'epoch':
        lr_step_per_epoch = True
    elif lr_step == 'batch':
        lr_step_per_epoch = False
    else:
        lr_step_per_epoch = True

    _scheduler = None
    if not lr_step_per_epoch:
        _scheduler = scheduler

    log = np.zeros([max_epoch, 2], dtype=np.float)

    for e in range(max_epoch):

        print('\nEpoch #{:d}'.format(e+1))

        log[e, 0] = train_single_epoch(net, optimizer, train_loader, noise_lvl,
                                       clip, lossfn, _scheduler)

        print('Train Loss: {:.5f}'.format(log[e, 0]))

        if scheduler != None and lr_step_per_epoch:
            scheduler.step()

        if validation != None:

            log[e, 1] = validate(net, validation, noise_lvl, clip, lossfn)

            print('Val Loss: {:.5f}'.format(log[e, 1]))

            if (checkpoint_dir != None) and (best_loss > log[e, 1]):
                best_loss = log[e, 1]
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint'+str(e+1)+'.pth')
                torch.save(net.state_dict(), checkpoint_path)
                print('Best Loss! Saved.')
            elif max_tolerance >= 0:
                tolerated += 1
                if tolerated > max_tolerance:
                    return log[0:e, :]

    return log

@torch.no_grad()
def inference(net, image):
    net.eval()
    transform = data.get_transform('test')
    x = transform(image)
    x = x.unsqueeze(0)
    x = x.to(net.device)
    x = net(x)
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose([1, 2, 0])
    x = x.squeeze(-1)
    x = np.clip(x, 0, 1) * 255
    return Image.fromarray(x.astype(np.uint8), 'L')
