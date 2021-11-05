
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

def train_single_epoch(net, optimizer, train_loader,
                       inputfn=data.inputfn(25),
                       lossfn=F.mse_loss,
                       scheduler=None):

    n_data = 0

    total_loss = 0.
    net.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=False, file=sys.stdout)

    for images in train_loader:
        optimizer.zero_grad()
        batch_size = images.size(0)
        images = images.to(net.device)
        x = inputfn(images)
        output = net(*x)
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
def validate(net, test_loader, inputfn=data.inputfn(25), lossfn=F.mse_loss):

    n_data = 0

    total_loss = 0.
    net.eval()

    pbar = tqdm(total=len(test_loader), position=0, leave=False, file=sys.stdout)

    for images in test_loader:
        batch_size = images.size(0)
        images = images.to(net.device)
        x = inputfn(images)
        output = net(*x)
        total_loss += lossfn(output, images).item() * batch_size
        n_data += batch_size
        pbar.update(1)

    tqdm.close(pbar)

    return total_loss / float(n_data)


def train(net, optimizer, max_epoch, train_loader,
          inputfn=data.inputfn(25), lossfn=F.mse_loss,
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

        log[e, 0] = train_single_epoch(net, optimizer, train_loader, inputfn, lossfn, _scheduler)

        print('Train Loss: {:.5f}'.format(log[e, 0]))

        if scheduler != None and lr_step_per_epoch:
            scheduler.step()

        if validation != None:

            log[e, 1] = validate(net, validation, inputfn, lossfn)

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
