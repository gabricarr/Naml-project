import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Custom dataset for the features in the csv files
class GTZANFeaturesDataset(Dataset):
    def __init__(self, features_file, transform=None, target_transform=None):
        """
        Args:
            features_file (string): Path to the csv file with features.
            img_dir (string): Directory with all the spectrogram images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # We drop the feature related to the length of the audio file
        self.features_frame = pd.read_csv(features_file).drop('length', axis=1)         
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.features_frame)


    def __getitem__(self, idx):
        # Load features
        features = self.features_frame.iloc[idx, 1:-1].to_numpy()
        #print(features.shape)
        features = features.astype('float')#.reshape(-1, 1)

        # Load label
        label = self.features_frame.iloc[idx, -1]


        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        # print(features.shape, label.shape)
        return features.float(), label
    


# Custom dataset form the images in the GTZAN dataset
class GTZANSpectogramDataset(Dataset):
    def __init__(self, features_file, img_dir, transform=None, target_transform=None):
        """
        Args:
            features_file (string): Path to the csv file with features.
            img_dir (string): Directory with all the spectrogram images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # We drop the feature related to the length of the audio file
        self.features_frame = pd.read_csv(features_file).drop('length', axis=1)         
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.features_frame)


    def __getitem__(self, idx):
        # Load image data
        while True:
            try:
                img_name = self.generate_gtzan_full_img_name(self.features_frame.iloc[idx, 0], self.img_dir)
                image_rgba = Image.open(img_name)
                break
            except:
                # We skip the image that we can't open
                #print(f"Error opening image {img_name}")  # 'jazz.00054.png' is missing
                idx += 1
        image = image_rgba.convert("L")    # We convert the img to greyscale

        # Load label
        label = self.features_frame.iloc[idx, -1]


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def generate_gtzan_full_img_name(self, original_img_name, img_dir):
        """ 
            The file name in the csv is 'reggae.00019.wav', 
            we neet to change it to '\\reggae\\reggae00019.png'
        """
        # We have to change the extension of the file
        img_file_name = original_img_name.replace('.wav', '.png')
        # To get the name of the folder we need to remove '.png' and the identifiers numbers (ex '.00000)'
        img_folder_name = img_file_name[:-10]    
        # Now we need to remove the '.' between 'reggae.00019'    
        img_file_name = img_file_name[:-10] + img_file_name[-9:]    
        # Now we can create the full path   
        img_name = os.path.join(img_dir, img_folder_name, img_file_name)

        return img_name