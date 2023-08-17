from typing import Callable
from random import randint
import os
import os.path
from os.path import exists
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
import random
from torchvision import transforms
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.distributed as dist
import torch.utils.data
import torchvision

import albumentations
from albumentations.pytorch import ToTensorV2

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]


def fft_spectrum(image):
    # Load the image and split it into color channels
    red_channel, green_channel, blue_channel = torch.chunk(image, 3, dim=-3)

    # Calculate the magnitude and phase spectrum for each color channel
    mag_spectrum = []
    phase_spectrum = []
    for channel in [red_channel, green_channel, blue_channel]:
        # Calculate the grayscale version of the channel
        gray_channel = transforms.Grayscale()(channel)

        # Calculate the DFT of the grayscale channel
        dft = torch.fft.fftn(gray_channel, dim=(-2, -1))

        # Calculate the magnitude and phase spectrum
        mag = torch.abs(dft)
        phase = torch.angle(dft)

        mag_spectrum.append(mag)
        phase_spectrum.append(phase)

    # Convert the spectra to logarithmic scale
    mag_spectrum = [torch.log10(mag + 1) for mag in mag_spectrum]
    phase_spectrum = [torch.log10(phase + 10) for phase in phase_spectrum]
    mag_spectrum_tensor = torch.stack(mag_spectrum, dim=1)
    phase_spectrum_tensor = torch.stack(phase_spectrum, dim=1)
    result = torch.cat((mag_spectrum_tensor, phase_spectrum_tensor), dim=1)
    return torch.squeeze(result, 0)


def fft_spectrum_gray(image):
    image = transforms.Grayscale()(image)
    dft = torch.fft.fftn(image, dim=(-2, -1))
    mag = torch.abs(dft)
    mag_spectrum = torch.log10(mag + 1)
    return mag_spectrum


def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv)  # head: image_path, label
    class_counts = dataframe.label.value_counts()
    sample_weights = [1 / class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler


# map_size is for PixBis
class TrainDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224), map_size=26):
        self.map_size = map_size
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # albumentations.RandomCrop(height=input_shape[0], width=input_shape[1])
            # albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            # ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0
        # map_x = torch.ones((self.map_size, self.map_size)) if label == 1 else torch.zeros((self.map_size, self.map_size))

        # pillow_image = Image.open(img_path)
        # image = np.array(pillow_image)
        image = self.composed_transformations(image=image)['image']
        image = transforms.ToTensor()(image)
        # fft = fft_spectrum(image)
        # fft = transforms.Normalize([0.56, 0.56, 0.56, 0.99, 0.99, 0.99],
        #                            [0.41, 0.41, 0.41, 0.08, 0.08, 0.08])(fft)
        # fft = fft_spectrum_gray(image)
        # fft = transforms.Resize(size=10)(fft)

        return {
            "images": image,
            # "fft": fft,
            "labels": torch.tensor(label, dtype=torch.float),
            # "map": map_x
        }


class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224), map_size=26):
        self.map_size = map_size
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            # albumentations.CenterCrop(height=input_shape[0], width=input_shape[1])
            # albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            # ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if label_str == 'bonafide' else 0
        # map_x = torch.ones((self.map_size, self.map_size)) if label == 1 else torch.zeros((self.map_size, self.map_size))

        # pillow_image = Image.open(img_path)
        # image = np.array(pillow_image)
        image = self.composed_transformations(image=image)['image']
        image = transforms.ToTensor()(image)
        # fft = fft_spectrum(image)
        # fft = transforms.Normalize([0.56, 0.56, 0.56, 0.99, 0.99, 0.99],
        #                            [0.41, 0.41, 0.41, 0.08, 0.08, 0.08])(fft)

        return {
            "images": image,
            # "fft": fft,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path,
            # "map": map_x
        }
