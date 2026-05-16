"""
Reference: The author of the main architecture of the code is:
https://github.com/LilLouis5/, Included:
    1. train (Added some of my features and modifications)
    2. evaluation (Added some of my features and modifications)
    3. main (Added some of my features and modifications)
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import albumentations as A
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, confusion_matrix

from Unet import Unet
from TransUnet import TransUnet
from utils import get_loaders, random_show_result, split_subset, MAPPING_INFO


IGNORED_IDX = {0, 1, 12, 13}    # unlabelled, asphalt, pole, vehicle

# Hyper params
CLASS_NUM = 19
BATCH_SIZE = 8     # TODO: adjust based on model used and CROPPED_INPUT_SIZE
NUM_EPOCH = 25
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_H = 1512
INPUT_W = 2016
CROPPED_INPUT_SIZE = 512


# set random seed
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataset used
TRAIN_SPLIT_RATIO = 0.3
TEST_SPLIT_RATIO = 0.1      # 213
TRAIN_DS = 'sub_train'
TEST_DS = 'sub_test'

# models    # TODO: set model type
# model_type = 'Unet'
model_type = 'TransUnet'

MODEL = None
if model_type == 'Unet':
    MODEL = Unet(
        in_channel=3,
        out_channel=CLASS_NUM
    ).to(DEVICE)

    LEARNING_RATE = 1e-3
    AMP_FLAG = True

elif model_type == 'TransUnet':
    MODEL = TransUnet(
        img_dim=CROPPED_INPUT_SIZE,
        in_channels=3,
        out_channels=128,
        head_num=4,
        mlp_dim=512,                    # TODO
        block_num=4,                    # TODO: most
        patch_dim=16,
        class_num=CLASS_NUM,
    ).to(DEVICE)

    LEARNING_RATE = 1e-4
    AMP_FLAG = False

# used model
MODEL_NAME = MODEL.__class__.__name__


def train(loader, model, loss_fn, optimizer, scaler=None):
    model.train()

    loop = tqdm(loader, desc='Training')
    total_loss = 0.0

    for idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.long().to(DEVICE)

        # use Automatic Mixed Precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                predict = model(data)
                loss = loss_fn(predict, target)

            # reset
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            predict = model(data)
            loss = loss_fn(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluation(loader, model, epoch, train_loss=None):
    loop = tqdm(loader, desc='Testing')

    jaccard_scores = [0] * CLASS_NUM
    class_counts = [0] * CLASS_NUM
    all_true_labels = []
    all_pred_labels = []

    model.eval()

    with torch.no_grad():
        for data, label in loop:
            data = data.to(DEVICE)
            label = label.long().to(DEVICE)
            batch_size, _, height, width = data.shape

            # use a sliding window to evaluate
            for i in range(0, height, CROPPED_INPUT_SIZE):
                for j in range(0, width, CROPPED_INPUT_SIZE):
                    window_data = data[:, :, i:i + CROPPED_INPUT_SIZE, j:j + CROPPED_INPUT_SIZE]
                    window_label = label[:, i:i + CROPPED_INPUT_SIZE, j:j + CROPPED_INPUT_SIZE]

                    # padding if the size after crop is not enough
                    if window_data.shape[2] != CROPPED_INPUT_SIZE or window_data.shape[3] != CROPPED_INPUT_SIZE:
                        pad_height = CROPPED_INPUT_SIZE - window_data.shape[2]
                        pad_width = CROPPED_INPUT_SIZE - window_data.shape[3]
                        window_data = torch.nn.functional.pad(window_data, (0, pad_width, 0, pad_height))
                        window_label = torch.nn.functional.pad(window_label, (0, pad_width, 0, pad_height))

                    preds = torch.argmax(model(window_data), dim=1)

                    for cls in range(CLASS_NUM):
                        pred_cls = (preds == cls).cpu().numpy().flatten()
                        true_cls = (window_label == cls).cpu().numpy().flatten()
                        score = jaccard_score(true_cls, pred_cls, zero_division=0)

                        jaccard_scores[cls] += score
                        class_counts[cls] += 1

                    all_true_labels.extend(window_label.cpu().numpy().flatten())
                    all_pred_labels.extend(preds.cpu().numpy().flatten())

        jaccard_per_class = [
            jaccard_scores[i] / class_counts[i] for i in range(CLASS_NUM) if i not in IGNORED_IDX
        ]
        miou = round(sum(jaccard_per_class) / (CLASS_NUM - len(IGNORED_IDX)), 4)

        with open('jaccard_scores.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + jaccard_per_class + [miou] + [train_loss])
            print('Jaccard Scores are saved to jaccard_scores.csv')

        print(f'Mean IoU: {miou}')

        # Compute and save confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(CLASS_NUM)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=MAPPING_INFO['classes'], yticklabels=MAPPING_INFO['classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        plt.savefig(f'{MODEL_NAME}_{epoch}_cm.png')
        plt.close()
        print(f'Confusion Matrix is saved as {MODEL_NAME}_{epoch}_cm.png')

    model.train()

    return jaccard_per_class, miou


def _evaluation(loader, model, epoch, train_loss=None):
    loop = tqdm(loader, desc='Testing')

    jaccard_scores = [0] * CLASS_NUM
    class_counts = [0] * CLASS_NUM

    model.eval()

    with torch.no_grad():
        for data, label in loop:
            data = data.to(DEVICE)
            label = label.long().to(DEVICE)

            # Resize the data and label to CROPPED_INPUT_SIZE
            data_resized = F.interpolate(
                data,
                size=(CROPPED_INPUT_SIZE, CROPPED_INPUT_SIZE),
            )

            preds_resized = torch.argmax(model(data_resized), dim=1)

            # Resize predictions back to original size
            preds = F.interpolate(
                preds_resized.unsqueeze(1).float(),
                size=(label.shape[1], label.shape[2]),
                mode='nearest'
            ).squeeze(1).long()

            for cls in range(CLASS_NUM):
                pred_cls = (preds == cls).cpu().numpy().flatten()
                true_cls = (label == cls).cpu().numpy().flatten()
                score = jaccard_score(true_cls, pred_cls, zero_division=1)

                jaccard_scores[cls] += score
                class_counts[cls] += 1

        jaccard_per_class = [
            jaccard_scores[i] / class_counts[i] for i in range(CLASS_NUM) if i not in IGNORED_IDX
        ]
        miou = round(sum(jaccard_per_class) / (CLASS_NUM - len(IGNORED_IDX)), 4)

        with open('jaccard_scores.csv', mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + jaccard_per_class + [miou] + [train_loss])
            print('Jaccard Scores are saved to jaccard_scores.csv')

        print(f'Mean IoU: {miou}')

    model.train()

    return jaccard_per_class, miou


def re_evaluation_specified_epoch(epoch):
    model_path = f'./Unet_{epoch}.pth'
    assert os.path.exists(model_path), f'Model {model_path} does not exist, train this epoch before evaluate it!'

    train_transform = A.Compose([
        # Consistency with the original paper
        A.RandomResizedCrop(height=CROPPED_INPUT_SIZE, width=CROPPED_INPUT_SIZE,
                            scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    train_loader, test_loader = get_loaders(
        train_dataset_type=TRAIN_DS,
        test_dataset_type=TEST_DS,
        train_transfrom=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    MODEL.load_state_dict(torch.load(f'./{MODEL_NAME}_{epoch}.pth'))
    evaluation(test_loader, MODEL, epoch)


def main():
    summary(MODEL, (3, CROPPED_INPUT_SIZE, CROPPED_INPUT_SIZE))
    train_transform = A.Compose([
        # Consistency with the original paper
        A.RandomResizedCrop(height=CROPPED_INPUT_SIZE, width=CROPPED_INPUT_SIZE,
                            scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    train_loader, test_loader = get_loaders(
        train_dataset_type=TRAIN_DS,
        test_dataset_type=TEST_DS,
        train_transfrom=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    if AMP_FLAG:
        scaler = torch.cuda.amp.GradScaler()    # Automatic Mixed Precision

    else:
        scaler = None

    # create csv to write evaluation data
    csv_file = 'jaccard_scores.csv'
    file_exists = os.path.exists(csv_file)
    with (open(csv_file, mode='a' if file_exists else 'w', newline='') as file):
        writer = csv.writer(file)

        if not file_exists:
            header = ['epoch'] + [MAPPING_INFO['classes'][idx] for idx in MAPPING_INFO['cidx'][:CLASS_NUM]] + \
                     ['miou'] + ['loss']
            writer.writerow(header)

        # write a new line for this training
        writer.writerow([
            f'{MODEL_NAME}, '
            f'Training set subset ratio: {TRAIN_SPLIT_RATIO}, '
            f'Testing set subset ratio: {TEST_SPLIT_RATIO}'
        ])

    for epoch in range(1, NUM_EPOCH + 1):
        print(f'\nCurrent epoch: {epoch}\n')

        cur_model_name = f'{MODEL_NAME}_{epoch}.pth'
        prev_model_name = f'{MODEL_NAME}_{epoch - 1}.pth'
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        cur_model_path = os.path.join(current_file_dir, cur_model_name)
        prev_model_path = os.path.join(current_file_dir, prev_model_name)

        if not os.path.exists(cur_model_path):
            if epoch > 1 and os.path.exists(prev_model_path):
                print(f'Loading model from epoch {epoch - 1} for training in epoch {epoch}')
                MODEL.load_state_dict(torch.load(prev_model_path))

            else:
                print(f'Starting new training in epoch {epoch}')

            train_loss = train(train_loader, MODEL, loss_fn, optimizer, scaler)

            torch.save(MODEL.state_dict(), cur_model_name)

            _, _ = evaluation(test_loader, MODEL, epoch, train_loss)

        else:
            print(f'Skip training in epoch: {epoch}')


if __name__ == '__main__':
    split_subset('train', TRAIN_SPLIT_RATIO)
    split_subset('test', TEST_SPLIT_RATIO)
    main()

    # epoch_num = np.random.randint(0, NUM_EPOCH)
    random_show_result(MODEL, DEVICE, MODEL_NAME, 15)

    # for i in range(1, 9):
    #     re_evaluation_specified_epoch(i)
    # re_evaluation_specified_epoch(15)
