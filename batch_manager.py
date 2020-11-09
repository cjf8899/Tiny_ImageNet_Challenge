import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class BatchManagerTinyImageNet(Dataset):
    def __init__(self, split='train',transform=None):
        assert split in ['train', 'val']
        self.transform = transform
        self._base_dir = './data/tiny-imagenet-200/'
        self._data_dir = os.path.join(self._base_dir, split)

        # list of (image_path, class_idx, class_code)
        self.image_paths_and_classes = []

        if split == 'train':
            self._class_codes = sorted([os.path.basename(class_dir) for class_dir in glob.glob(os.path.join(self._data_dir, '*'))])
            for class_code in self._class_codes:
                class_image_paths = glob.glob(os.path.join(self._data_dir, class_code, '*/*'))
                for class_image_path in class_image_paths:
                    if os.path.splitext(os.path.basename(class_image_path))[-1].lower() in ['.jpeg', '.jpg', '.png']:
                        self.image_paths_and_classes.append((class_image_path, class_code))
        elif split == 'val':
            self._class_codes = []
            with open(os.path.join(self._data_dir, 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_tokens = line.strip().split("\t")
                    im_name, class_code = line_tokens[0], line_tokens[1]
                    self.image_paths_and_classes.append((os.path.join(self._data_dir, 'images', im_name), class_code))
                    self._class_codes.append(class_code)
            self._class_codes = sorted(list(set(self._class_codes)))

    def __getitem__(self, sample_idx):
        image_path, class_code = self.image_paths_and_classes[sample_idx]
        class_idx = self._class_codes.index(class_code)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
#         image = np.transpose(cv2.imread(image_path), axes=(2,0,1)).astype(np.float32)
        
        return image, class_idx

    def __len__(self):
        return len(self.image_paths_and_classes)