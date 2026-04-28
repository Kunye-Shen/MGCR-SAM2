import torch
import random
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class ToTensor(object):
    def __call__(self, data):
        image, label, name = data['image'], data['label'], data['name']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label), 'name': name}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label, name = data['image'], data['label'], data['name']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC), 'name': name}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label, name = data['image'], data['label'], data['name']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label), 'name': name}

        return {'image': image, 'label': label, 'name': name}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label, name = data['image'], data['label'], data['name']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label), 'name': name}

        return {'image': image, 'label': label, 'name': name}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, name = sample['image'], sample['label'], sample['name']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label, 'name': name}

def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)

    out = (x - MIN) / (MAX - MIN)

    return out

class DefectDataset(Dataset):
    def __init__(self, data_path, transform, mode='train'):
        if mode == 'train':
            self.image_list = glob(data_path + f"{mode}/Img/*")
            self.label_list = glob(data_path + f"{mode}/GT/*")
        else:
            self.image_list = glob(data_path + f"{mode}/Img/*")
            self.label_list = glob(data_path + f"{mode}/GT/*")
        self.image_list.sort()
        self.label_list.sort()
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_dir = self.image_list[index]
        label_dir = self.label_list[index]

        image = Image.open(image_dir).convert('RGB')

        label = Image.open(label_dir).convert('L')

        name = self.image_list[index]

        sample = {'image':image, 'label':label, 'name':name}
        sample = self.transform(sample)

        return sample