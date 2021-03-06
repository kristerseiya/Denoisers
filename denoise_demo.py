
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import utils
import dnsr
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--noiselvl', type=float, default=25)
parser.add_argument('--denoiser', type=str, default='dncnn')
parser.add_argument('--weights', type=str, default='dsnr/DnCNN/dncnn50.pth')
args = parser.parse_args()

img = Image.open(args.image).convert('L')
img = np.array(img) / 255.
noisy = img + np.random.normal(size=img.shape) * args.noiselvl / 255.

if args.denoiser == 'dncnn':
    net = dnsr.DnCNN(args.weights)
    recon = net(noisy)
elif args.denoiser == 'cdncnn':
    net = dnsr.cDnCNN(args.weights)
    net.set_param(args.noiselvl/255.)
    recon = net(noisy)
elif args.denoiser == 'bm3d':
    bm3d= dnsr.BM3D()
    bm3d.set_param(args.noiselvl/255.)
    recon = bm3d(noisy)
elif args.denoiser == 'tv':
    tvprox = dnsr.ProxTV(lambd=(args.noiselvl/255.*7)**2)
    recon = tvprox(noisy)

mse, psnr = utils.compute_mse(img, recon, scale=1)
ssim = utils.ssim(img, recon, scale=1).mean()
msssim = utils.msssim(img, recon, scale=1).mean()
print('MSE: {:.5f}'.format(mse))
print('PSNR: {:.5f}'.format(psnr))
print('SSIM: {:.5f}'.format(ssim))
print('MS-SSIM: {:.5f}'.format(msssim))

utils.stackview([img, noisy, recon], width=20, method='Pillow')
