"""
Reference: The author of the main architecture of the code is:
https://github.com/LilLouis5/, Included:
    1. get_loaders
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from dataset import WildScenesDataset


TEST_BATCH_SIZE = 1
CROPPED_INPUT_SIZE = 512

MAPPING_INFO = {
    'classes': (
        'unlabelled',
        'asphalt',
        'dirt',
        'mud',
        'water',
        'gravel',
        'other-terrain',
        'tree-trunk',
        'tree-foliage',
        'bush',
        'fence',
        'structure',
        'pole',
        'vehicle',
        'rock',
        'log',
        'other-object',
        'sky',
        'grass',
    ),
    'palette': [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (230, 25, 75),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (250, 190, 190),
        (0, 0, 128),
        (128, 128, 128),
    ],
    'cidx': [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18
    ]
}

# root manegement
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../'))


def split_subset(dataset_type, slpit_factor=0.1):
    # Load the dataset
    file_path = os.path.join(ROOT_DIR, 'data/splits/opt2d', f'{dataset_type}.csv')
    subset_file_path = os.path.join(ROOT_DIR, 'data/splits/opt2d', f'sub_{dataset_type}.csv')

    data = pd.read_csv(file_path)

    # Randomly select some of the data
    subset_data = data.sample(frac=slpit_factor, random_state=0)

    # Save the subset to a new CSV file
    subset_data.to_csv(subset_file_path, index=False)


def get_loaders(train_dataset_type, test_dataset_type,
                train_transfrom, test_transform,
                batch_size, num_workers, pin_memory=True):
    train_ds = WildScenesDataset(train_dataset_type, transform=train_transfrom)
    test_ds = WildScenesDataset(test_dataset_type, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def convert_to_rgb(label):
    palette = MAPPING_INFO['palette']
    height, width = label.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in enumerate(palette):
        rgb_image[label == class_idx] = color

    return rgb_image


def random_show_result(model, device, model_name, epoch_num, random_idx=None):
    csv_path = os.path.join(ROOT_DIR, f'data/splits/opt2d/val.csv')
    base_path = os.path.join(ROOT_DIR, 'data/WildScenes/')

    data = pd.read_csv(csv_path)
    data_length = len(data)

    # random select an image and its label
    if random_idx is None:
        random_idx = np.random.randint(0, data_length + 1)

    image_paths = data['im_path'].tolist()
    label_paths = data['label_path'].tolist()

    image_path = image_paths[random_idx]
    label_path = label_paths[random_idx]

    image_path = base_path + image_path
    label_path = base_path + label_path

    image = np.array(Image.open(image_path).convert('RGB'))
    label = np.array(Image.open(label_path).convert('L'))

    ori_h, ori_w = image.shape[:2]

    image_resized = np.array(Image.fromarray(image).resize((512, 512), Image.BILINEAR))
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dim

    # load model
    model_path = f'{model_name}_{epoch_num}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = torch.argmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy()

    # Resize prediction back to original size
    pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((ori_w, ori_h), Image.NEAREST))

    # convert to rgb
    label_rgb = convert_to_rgb(label)
    pred_rgb = convert_to_rgb(pred)

    # Display images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Label')
    plt.imshow(label_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(pred_rgb)
    plt.axis('off')

    plt.show()


def generate_directory_structure(startpath='..'):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')


generate_directory_structure()
