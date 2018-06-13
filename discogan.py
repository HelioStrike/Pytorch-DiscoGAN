import argparse
import os
import numpy as np
import math
import itertools
import sys
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

def main():
    # Create sample and checkpoint directories
    os.makedirs('images/edges2shoes_small', exist_ok=True)
    os.makedirs('saved_models/edges2shoes_small', exist_ok=True)

    #init weights
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if(classname.find('Conv') != -1):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    #loss functions
    adversarial_loss = nn.MSELoss()         #To check how well our generators are doing against our discriminators and vice-versa
    cycle_loss = nn.L1Loss()                #Reconstruction loss
    pixelwise_loss = nn.L1Loss()            #To compare fake images with real images

    #Checking for cuda
    cuda = True if torch.cuda.is_available() else False

    #declaring nets
    G_AB = GeneratorUNet()
    G_BA = GeneratorUNet()
    D_A = Discriminator()
    D_B = Discriminator()

    if(cuda):
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        adversarial_loss.cuda()
        cycle_loss.cuda()
        pixelwise_loss.cuda()

    #Initializing network weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    #defining optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #transforms
    tfms = [
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    #dataloaders
    trainloader = DataLoader(ImageDataset(r"D:\myStuff\pytorch\tuts\1\data\edges2shoes\edge2shoes_small", transforms_=tfms, mode='train'), batch_size=16,
                                shuffle=True, num_workers=4)
    valloader = DataLoader(ImageDataset(r"D:\myStuff\pytorch\tuts\1\data\edges2shoes\edge2shoes_small", transforms_=tfms, mode='val'), batch_size=4,
                                shuffle=True, num_workers=4)


    #save results for every few epochs
    def sample_images(epoch_num):
        imgs = next(iter(valloader))
        real_A = imgs['A'].cuda()
        fake_B = G_AB(real_A)
        real_B = imgs['B'].cuda()
        fake_A = G_BA(real_B)
        fin = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)
        save_image(fin, 'images/edges2shoes_small/%s.png' % (epoch_num), nrow=8, normalize=True)
        print("Saving images...")

    for epoch in range(50):
        epoch_G_loss = 0
        epoch_D_loss = 0
        for i, batch in enumerate(trainloader):
            real_A = batch['A'].cuda()
            real_B = batch['B'].cuda()

            valid = torch.ones(real_A.size(0), 1, requires_grad=False).cuda()
            fake = torch.zeros(real_B.size(0), 1, requires_grad=False).cuda()


            #############################
            # Train Generators
            #############################
            optimizer_G.zero_grad()

            #GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2

            #pixelwise translation loss
            loss_pixelwise = (pixelwise_loss(fake_B, real_B) + pixelwise_loss(fake_A, real_A))/2

            #cycle loss
            loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
            loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B)/2

            #total loss
            loss_G = loss_GAN + loss_pixelwise + loss_cycle
            loss_G.backward()
            optimizer_G.step()

            epoch_G_loss += loss_G.data

            #############################
            # Train Discriminators
            #############################
            optimizer_D_A.zero_grad()

            loss_real = adversarial_loss(D_A(real_A), valid)
            loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)

            loss_D_A = (loss_real + loss_fake)/2
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            loss_real = adversarial_loss(D_B(real_B), valid)
            loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)

            loss_D_B = (loss_real + loss_fake)/2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = 0.5*(loss_D_A + loss_D_B)
            epoch_D_loss += loss_D.data

        print("Epoch", epoch, "Generator loss:", epoch_G_loss, "Discriminator loss:", epoch_D_loss)
        sample_images(epoch)

        if epoch % 2 == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), 'saved_models/edges2shoes_small/G_AB.pth')
            torch.save(G_BA.state_dict(), 'saved_models/edges2shoes_small/G_BA.pth')
            torch.save(D_A.state_dict(), 'saved_models/edges2shoes_small/D_A.pth')
            torch.save(D_B.state_dict(), 'saved_models/edges2shoes_small/D_B.pth')

if __name__=='__main__':
    main()
