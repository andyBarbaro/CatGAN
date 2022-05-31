"""
This project is used to produce 64x64 pixel images of cats' faces given a dataset
of nearly 16,000 images of cats' faces. A GAN is developed as two seperate subclasses
of the torch.nn.Module class.
The code for the neural network was written by Andy Barbaro with help from Jovian and Aakash Rao's
phenomenal introduction to GANs which helped me implement some of the finer details of my model
and was responsible for the code that cleanly outputs the results of my GAN.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import save_image
import random
import os
import time
from matplotlib import pyplot as plt

#used for nomralizing and denormalizing images to achienve pixel values for RGB
#pixels between values -1 and 1 (useful due to output of discriminator)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
#set size of noise to be passed into generator
noise_size = 128
#set random noise to compare improvements over each epoch
set_noise = torch.randn(64, noise_size, 1, 1)


def denormalize(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

#From Jovian tutorial on GANs
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denormalize(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    #plt.show()

#From Jovian tutorial on GANs
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

#From Jovian tutorial on GANs
def save_samples(index, latent_tensors, gen):
    fake_images = gen.forward(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denormalize(fake_images), os.path.join('./cats', fake_fname), nrow=8)
    print('Saving', fake_fname)


class Discriminator(nn.Module):
    # paramater in_outs is a list of number of in channels and out channels per cnn layer
    def __init__(self, in_outs, kernel, skip, padding_size):
        super().__init__()
        self.l_relu = nn.LeakyReLU(0.2)
        # define cnn layers
        self.cnn1 = nn.Conv2d(in_outs[0][0], in_outs[0][1], kernel, skip, padding_size)
        self.batch_norm1 = nn.BatchNorm2d(in_outs[0][1])
        self.cnn2 = nn.Conv2d(in_outs[1][0], in_outs[1][1], kernel, skip, padding_size)
        self.batch_norm2 = nn.BatchNorm2d(in_outs[1][1])
        self.cnn3 = nn.Conv2d(in_outs[2][0], in_outs[2][1], kernel, skip, padding_size)
        self.batch_norm3 = nn.BatchNorm2d(in_outs[2][1])
        self.cnn4 = nn.Conv2d(in_outs[3][0], in_outs[3][1], kernel, skip, padding_size)
        self.batch_norm4 = nn.BatchNorm2d(in_outs[3][1])
        self.cnn5 = nn.Conv2d(in_outs[4][0], in_outs[4][1], kernel)
        self.flat = nn.Flatten()
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x = self.l_relu(self.batch_norm1(self.cnn1(data)))
        x = self.l_relu(self.batch_norm2(self.cnn2(x)))
        x = self.l_relu(self.batch_norm3(self.cnn3(x)))
        x = self.l_relu(self.batch_norm4(self.cnn4(x)))
        x = self.sig(self.flat(self.cnn5(x)))
        return x


class Generator(nn.Module):
    # paramater in_outs is a list of number of in channels and out channels per cnn layer
    def __init__(self, in_outs, kernel, skip, padding_size):
        super().__init__()
        self.relu = nn.ReLU()
        # define cnn layers
        self.rev_cnn1 = nn.ConvTranspose2d(in_outs[0][0], in_outs[0][1], kernel, skip)
        self.batch_norm1 = nn.BatchNorm2d(in_outs[0][1])
        self.rev_cnn2 = nn.ConvTranspose2d(in_outs[1][0], in_outs[1][1], kernel, skip, padding_size)
        self.batch_norm2 = nn.BatchNorm2d(in_outs[1][1])
        self.rev_cnn3 = nn.ConvTranspose2d(in_outs[2][0], in_outs[2][1], kernel, skip, padding_size)
        self.batch_norm3 = nn.BatchNorm2d(in_outs[2][1])
        self.rev_cnn4 = nn.ConvTranspose2d(in_outs[3][0], in_outs[3][1], kernel, skip, padding_size)
        self.batch_norm4 = nn.BatchNorm2d(in_outs[3][1])
        self.rev_cnn5 = nn.ConvTranspose2d(in_outs[4][0], in_outs[4][1], kernel, skip, padding_size)
        self.batch_norm5 = nn.BatchNorm2d(in_outs[4][1])
        self.tanh = nn.Tanh()


    def forward(self, data):
        x = self.relu(self.batch_norm1(self.rev_cnn1(data)))
        x = self.relu(self.batch_norm2(self.rev_cnn2(x)))
        x = self.relu(self.batch_norm3(self.rev_cnn3(x)))
        x = self.relu(self.batch_norm4(self.rev_cnn4(x)))
        x = self.tanh(self.rev_cnn5(x))
        return x


def train_disc(disc, gen, optimizer, loss_func, real_images):

    optimizer.zero_grad()

    #train discriminator on real cat images from dataset
    real_labels = torch.ones(real_images.size(0), 1)
    real_preds = disc.forward(real_images)
    loss = loss_func(real_preds, real_labels)
    real_preds = real_preds.detach()

    #train discriminator on fake cat images from generator
    fake_labels = torch.zeros(real_images.size(0), 1)
    random_noise_data = torch.randn(real_images.size(0), noise_size, 1, 1)
    fake_images = gen.forward(random_noise_data)
    fake_preds = disc.forward(fake_images)
    print(fake_preds.shape)
    print(fake_labels.shape)
    print("++++")
    loss += loss_func(fake_preds, fake_labels)

    fake_images = fake_images.detach()
    fake_preds = fake_preds.detach()
    #backpropogate and update weights
    loss.backward()
    optimizer.step()

    return loss


def train_gen(gen, disc, optimizer, loss_func, batch_size):

    optimizer.zero_grad()

    #generate fake images and classify them as real images for discriminator
    #effectively trains generator against the discriminator
    false_labels = torch.ones(batch_size, 1)
    random_noise_data = torch.randn(batch_size, noise_size, 1, 1)
    fake_images = gen.forward(random_noise_data)
    preds = disc.forward(fake_images)
    loss = loss_func(preds, false_labels)

    fake_images = fake_images.detach()
    preds = preds.detach()
    #backpropogate and update weights
    loss.backward()
    optimizer.step()

    return loss


if __name__ == '__main__':
    #convert images into tensors normalize images before storing in image folder
    cats_if = ImageFolder("./cats", transform=T.Compose([T.ToTensor(), T.Normalize(*stats)]))
    cats_dl = DataLoader(cats_if, batch_size=128, shuffle=True)

    # in and out channel values for discriminator
    in_outs_disc = [(3,64), (64,128), (128,256), (256,512), (512,1)]
    # in and out channel values for generator
    in_outs_gen = [(128,512), (512,256), (256,128), (128,64), (64,3)]

    gen = Generator(in_outs_gen, kernel=4, skip=2, padding_size=1)
    disc = Discriminator(in_outs_disc, kernel=4, skip=2, padding_size=1)

    disc_optim = optim.Adam(disc.parameters(), lr=0.001, betas=(0.5, 0.999))
    disc_bce_loss = nn.BCELoss()
    gen_optim = optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
    gen_bce_loss = nn.BCELoss()

    random_noise_data = torch.randn(noise_size, 1, 1)
    print(random_noise_data)


    batch_size = 128
    num_epochs = 24
    for epoch in range(num_epochs):
        total_d_loss = 0
        total_g_loss = 0
        for image_batch, _ in cats_dl:
            d_loss = train_disc(disc, gen, disc_optim, disc_bce_loss, image_batch)
            g_loss = train_gen(gen, disc, gen_optim, gen_bce_loss, batch_size)
            total_d_loss += d_loss
            total_g_loss += g_loss

        print("Epoch %i -> Discriminator Loss: %.3f  Generator Loss: %.3f" % (epoch, total_d_loss, total_g_loss))

        save_samples(epoch, set_noise, gen)
