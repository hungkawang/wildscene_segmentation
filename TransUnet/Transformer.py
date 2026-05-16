import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score


class CustomConvLayer(nn.Module):
    """
    Use Gaussian kernel to scale original image to a small size
    """
    def __init__(self, kernel_size=3):
        super(CustomConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.gaussian_kernel = self.create_gaussian_kernel()

    def create_gaussian_kernel(self, sigma=1.0):
        k = cv2.getGaussianKernel(self.kernel_size, sigma)
        gaussian_kernel = np.outer(k, k)
        gaussian_kernel = torch.tensor(gaussian_kernel, dtype=torch.float32)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

        return gaussian_kernel

    def forward(self, x):
        out = nn.functional.conv2d(x, self.gaussian_kernel.repeat(3, 1, 1, 1), stride=self.stride, groups=3)

        return out


class PatchEmbedding(nn.Module):
    """
    Use a window (stride = kernel_size) to perform convolution on the original image
    to segement the image into small pathces

    Reference: https://github.com/LilLouis5/Vision-Transformer/blob/main/model.py
    """
    def __init__(self, in_channels, patch_size, num_patches, gaussian_kernel_size=3, dropout=0.001):
        super(PatchEmbedding, self).__init__()
        embed_dim = (patch_size ** 2) * in_channels
        self.custom_conv = CustomConvLayer(gaussian_kernel_size)
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # use CustomConvLayer to scale
        x = self.custom_conv(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer

    Reference: https://github.com/LilLouis5/Vision-Transformer/blob/main/model.py
    """
    def __init__(self, in_channels, patch_size, dropout,
                 num_heads, activation, num_encoders, gaussian_kernel_size):
        embed_dim = (patch_size ** 2) * in_channels
        self.patch_size = patch_size
        super(VisionTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # Decoder: Transposed Convolution layers for upsampling
        self.decoder_blocks = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.ConvTranspose2d(embed_dim, CLASS_NUM, kernel_size=gaussian_kernel_size, stride=gaussian_kernel_size)
            )

    def forward(self, x):
        x = self.encoder_blocks(x)

        # Remove cls token
        x = x[:, 1:, :].transpose(1, 2)

        # reshape
        batch_size, embed_dim, num_patches = x.shape
        x = x.reshape(batch_size, INPUT_H // GAUSSIAN_KERNEL_SIZE // PATCH_SIZE,
                      INPUT_W // GAUSSIAN_KERNEL_SIZE // PATCH_SIZE, CHANNEL_NUM * PATCH_SIZE ** 2)
        x = x.permute(0, 3, 1, 2)

        # Apply transposed convolutions to upsample to original image size
        x = self.decoder_blocks(x)

        return x


def get_csv_length(dataset_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../'))
    csv_path = os.path.join(root_dir, f'data/splits/opt2d/{dataset_type}.csv')

    with open(csv_path, 'r') as f:
        data_len = sum(1 for row in f) - 1    # sum - header

    return data_len


def _process_image(image_path, patch_size, gaussian_kernel_size, dropout):
    """
    # Use Gaussian kernel to resize image
    Use PatchEmbedding to split image into several small patches
    """
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.array(input_image)

    input_image_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # calculate patch_nums
    h, w = input_image_tensor.shape[2] // gaussian_kernel_size, input_image_tensor.shape[3] // gaussian_kernel_size
    num_patches = (h // patch_size) * (w // patch_size)

    # init PatchEmbedding
    patch_embedding = PatchEmbedding(
        in_channels=3,
        patch_size=patch_size,
        num_patches=num_patches,
        gaussian_kernel_size=gaussian_kernel_size,
        dropout=dropout,
    )

    with torch.no_grad():
        embedded_image_tensor = patch_embedding(input_image_tensor)

    '''
    exchange dim:
    (batch, patch_h * patch_w, patch_nums * channels) ->
    (batch, patch_nums * channels, patch_h * patch_w)
    '''
    convolved_image = embedded_image_tensor.squeeze(0).numpy()

    return convolved_image


def process_image(dataset_type, patch_size, gaussian_kernel_size, batch_size, dropout, start_index=0, test_flag=False):
    """
    Batch processing of images
    :param dataset_type: train, test or val dataset csv file
    :param patch_size: kernel size of conv kernel in PatchEmbbeding
    :param gaussian_kernel_size: gaussian kernel size of conv kernel in custome conv layer
    :param batch_size: Number of images batch processed
    :param dropout: Dropout rate
    :param start_index: start batch in csv
    :param test_flag: if True, show 1 pair of image (input, conv, label)
    :return:
        1. processed images (batch, (patchsize ** 2) * c, patch_h * patch_w)
        2. label images (batch, lb_h, lb_w)
    """
    assert dataset_type in ['train', 'test', 'val', 'sub_train', 'sub_test'], \
        'param dataset_type error: should be train, test or val'
    assert type(gaussian_kernel_size) is int and gaussian_kernel_size > 0, \
        'param gaussian_kernel_size error: should be a positive integer'
    assert type(patch_size) is int and patch_size > 0, 'param patch_size error: should be a positive integer'
    assert type(batch_size) is int and batch_size > 0, 'param batch_size error: should be a positive integer'

    # path management
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../'))
    csv_path = os.path.join(root_dir, f'data/splits/opt2d/{dataset_type}.csv')
    base_path = os.path.join(root_dir, 'data/WildScenes')

    data = pd.read_csv(csv_path)
    assert type(start_index) is int or start_index > len(data), \
        f'param start_index is beyond the length of the dataset: {start_index} / {len(data)}'

    # Select the batch data based on start_index and batch_size
    batch_data = data.iloc[start_index:start_index + batch_size]

    # use a list to store single data as a batch
    processed_images = []
    labels = []

    for _, row in batch_data.iterrows():
        # image_id = row['id']
        im_path = os.path.join(base_path, row['im_path'])
        label_path = os.path.join(base_path, row['label_path'])

        convolved_image = _process_image(im_path, patch_size, gaussian_kernel_size, dropout)
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        processed_images.append(convolved_image)
        labels.append(label_image)

        # Test code, to show image
        if test_flag:
            input_image = cv2.imread(im_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = np.array(input_image)

            # Remove position embedding
            convolved_image = convolved_image[:, :-1]

            # Reshape to (num_patches, patch_size, patch_size, channels)
            h, w, _ = input_image.shape
            h_patches = h // gaussian_kernel_size // patch_size
            w_patches = w // gaussian_kernel_size // patch_size

            reshaped_image = convolved_image.reshape(h_patches, w_patches, patch_size, patch_size, 3)
            reshaped_image = reshaped_image.transpose(0, 2, 1, 3, 4)
            reshaped_image = reshaped_image.reshape(h_patches * patch_size, w_patches * patch_size, 3)

            fig, axs = plt.subplots(1, 3, figsize=(12, 6))

            axs[0].imshow(input_image)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(reshaped_image)
            axs[1].set_title('Convolved Image')
            axs[1].axis('off')

            axs[2].imshow(label_image)
            axs[2].set_title('Label Image')
            axs[2].axis('off')

            plt.show()

            print('Run test code done!')

            return np.zeros((0, 3, 0, 0)), np.zeros((0, 1, 0, 0))

    labels = np.stack(labels, axis=0)

    processed_images_np = np.array(processed_images)
    processed_images_tensor = torch.tensor(processed_images_np).float() / 255.0
    labels_tensor = torch.tensor(labels).long()

    return processed_images_tensor, labels_tensor


def evaluate_models(dataset_name):
    model_files = sorted([f for f in os.listdir() if f.startswith('model_') and f.endswith('.pth')],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    eval_data_length = get_csv_length(dataset_name)

    for model_file in tqdm(model_files):
        eval_epoch = int(model_file.split('_')[1].split('.')[0])

        eval_model = VisionTransformer(
            CHANNEL_NUM,
            PATCH_SIZE,
            DROPOUT,
            HEAD_NUM,
            ACTIVATION,
            ENCODERS_NUM,
            GAUSSIAN_KERNEL_SIZE
        ).to(DEVICE)

        eval_model.load_state_dict(torch.load(model_file))
        eval_model.eval()

        total_iou_per_class = None
        total_batches = 0

        eval_start_index = 0

        while eval_start_index < eval_data_length:
            with torch.no_grad():
                val_batch_data, val_batch_labels = process_image(
                    dataset_name, PATCH_SIZE, GAUSSIAN_KERNEL_SIZE, BATCH_SIZE, DROPOUT, eval_start_index, False
                )

                val_batch_data = val_batch_data.to(DEVICE)
                val_batch_labels = val_batch_labels.to(DEVICE)

                eval_outputs = model(val_batch_data)
                eval_preds = torch.argmax(eval_outputs, dim=1)

                batch_iou_per_class = jaccard_score(val_batch_labels.cpu().numpy().flatten(),
                                                    eval_preds.cpu().numpy().flatten(), average=None)

                if total_iou_per_class is None:
                    total_iou_per_class = batch_iou_per_class

                else:
                    total_iou_per_class += batch_iou_per_class

                total_batches += 1

                eval_start_index += BATCH_SIZE

        avg_iou_per_class = total_iou_per_class / total_batches
        miou = avg_iou_per_class.mean()

        print(f'Model {model_file}, mIoU: {miou:.4f}')
        for class_idx, class_iou in enumerate(avg_iou_per_class):
            print(f'Class {class_idx} IoU: {class_iou:.4f}')


def train(train_dataset_type):
    # load model params or create a new one
    model_files = [f for f in os.listdir('.') if f.startswith('model_') and f.endswith('.pth')]
    if model_files:
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_model_file = model_files[-1]
        start_epoch = int(latest_model_file.split('_')[1].split('.')[0])
        print(f'Found existing model: {latest_model_file}, resuming from epoch {start_epoch}')

        model.load_state_dict(torch.load(latest_model_file))

    else:
        start_epoch = 0
        print('No existing model found, starting training from scratch')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    data_length = get_csv_length(train_dataset_type)

    # Train
    for epoch in tqdm(range(start_epoch, EPOCHES), desc='Training Epoch'):
        start_index = 0
        model.train()
        running_loss = 0.0

        while start_index < data_length:
            train_batch_data, train_batch_labels = process_image(
                train_dataset_type, PATCH_SIZE, GAUSSIAN_KERNEL_SIZE, BATCH_SIZE, DROPOUT, start_index, False
            )

            train_batch_data = train_batch_data.to(DEVICE)
            train_batch_labels = train_batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(train_batch_data)
            loss = criterion(outputs, train_batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * train_batch_data.size(0)
            start_index += BATCH_SIZE

        epoch_loss = running_loss / data_length
        print(f'Epoch {epoch + 1}/{EPOCHES}, Loss: {epoch_loss:.4f}')

        # save model
        model_save_path = f'model_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')


if __name__ == '__main__':
    # Hyper params
    INPUT_H = 1512
    INPUT_W = 2016
    CHANNEL_NUM = 3
    GAUSSIAN_KERNEL_SIZE = 9
    PATCH_SIZE = 8
    PATCH_NUM = (INPUT_H // PATCH_SIZE) * (INPUT_W // PATCH_SIZE)
    DROPOUT = 0.001
    HEAD_NUM = 8
    ACTIVATION = 'relu'
    ENCODERS_NUM = 6
    CLASS_NUM = 19      # actually 15
    T_CONV_2_PADDING = 0

    LEARNING_RATE = 1e-2
    EPOCHES = 25
    BATCH_SIZE = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model init
    model = VisionTransformer(
        CHANNEL_NUM,
        PATCH_SIZE,
        DROPOUT,
        HEAD_NUM,
        ACTIVATION,
        ENCODERS_NUM,
        GAUSSIAN_KERNEL_SIZE
    ).to(DEVICE)

    '''
    (INPUT_H, INPUT_W)
    -> (INPUT_H // GAUSSIAN_KERNEL_SIZE, INPUT_W // GAUSSIAN_KERNEL_SIZE)
    -> (INPUT_H // GAUSSIAN_KERNEL_SIZE // PATCH_SIZE, INPUT_W // GAUSSIAN_KERNEL_SIZE // PATCH_SIZE)
    embed_dim = (patch_size ** 2) * in_channels
    '''

    # train('sub_train')

    evaluate_models('sub_test')
