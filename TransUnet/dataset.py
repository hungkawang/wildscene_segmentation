"""
Reference: The author of the main architecture of the code is:
https://github.com/LilLouis5/, Included:
    1. WildScenesDataset (Added some of my features and modifications)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WildScenesDataset(Dataset):
    def __init__(self, dataset_type, transform=None):
        assert dataset_type in ['train', 'test', 'val', 'sub_train', 'sub_test'], \
            'param dataset_type error: should be train, test or val'

        self.transform = transform

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '../'))
        csv_path = os.path.join(root_dir, f'data/splits/opt2d/{dataset_type}.csv')
        base_path = os.path.join(root_dir, 'data/WildScenes/')

        data = pd.read_csv(csv_path)
        image_paths = data['im_path'].tolist()
        label_paths = data['label_path'].tolist()

        self.image_paths = [base_path + path for path in image_paths]
        self.label_paths = [base_path + path for path in label_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image = np.array(Image.open(image_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('L'))

        if self.transform:
            augmentation = self.transform(image=image, mask=label)
            image = augmentation['image']
            label = augmentation['mask']

        return image, label
