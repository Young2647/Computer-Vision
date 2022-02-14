from cv2 import transform
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image

class NYUdataset(Dataset) :
    def __init__(self, data_path, transform = None) :
        super(NYUdataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)        #normalize parameters

    def __getitem__(self, index):
        str_index = '0'
        for i in range(5 - len(str(index))) :
            str_index += '0'
        str_index += str(index) #00xxxx

        img_name = self.data_path + '{}_color.jpg'.format(str_index)
        depth_name = self.data_path + '{}_depth.png'.format(str_index)

        image = Image.open(img_name)
        depth = Image.open(depth_name)

        if self.transform == None :
            depth_array = np.array(depth)
            depth_array = depth_array.astype(np.float) / 255.*10.
            depth = Image.fromarray(depth_array)
            image = image.resize((304,228))
            depth = depth.resize((152,114))
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(self.mean, self.std)(image)
            depth = transforms.ToTensor()(depth)
        else :
            depth_array = np.array(depth)
            depth_array = depth_array.astype(np.float) / 1000.
            depth = Image.fromarray(depth_array)
            image = image.resize((304,228))
            depth = depth.resize((304,228))
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(self.mean, self.std)(image)
            depth = transforms.ToTensor()(depth)
        sample = {'image': image, 'depth': depth, 'name' : str_index}
        return sample
    
    def __len__(self) :
        return len(os.listdir(self.data_path))//2

def load_data(_datapath, batch_size, transform = None) :
    nyu_data = NYUdataset(_datapath, transform)
    _dataloader = DataLoader(nyu_data, batch_size, shuffle=True, num_workers=0, pin_memory=False)

    return _dataloader
