
# README

## Project Overview

This project is developed using Python 3.9.13 and includes several deep learning-related libraries. Please refer to the `requirements.txt` file in the root directory for specific version information of the libraries.

The project aims to use the `TransUnet` model for image segmentation on the `WildScenes` dataset. A simple demo is provided in the project, demonstrating how to use the pre-trained model `TransUnet_15.pth` for prediction on a small number of images, as well as how to train and test the model on a larger dataset.

## Directory Structure

When testing the code, the project directory structure should be as follows:

```
├─data
│  ├─splits
│  │  └─opt2d
│  └─WildScenes
│      └─WildScenes2d
│          ├─K-01
│          ├─K-03
│          ├─V-01
│          ├─V-02
│          └─V-03
├─metadata
└─Models_Sam
    ├─dataset.py
    ├─trainer.py
    ├─TranUnet.py
    ├─Unet.py
    ├─utils.py
    ├─VisionTransformer.py
    ├─Demo.ipynb
    ├─requirements.txt
    ├─run.bat
    └─TransUnet_15.pth
```

- The `data` and `metadata` folders are from the `WildScenes` dataset.
- The `Models_Sam` folder should contain the following files and scripts:
  - `dataset.py`
  - `trainer.py`
  - `TranUnet.py`
  - `Unet.py`
  - `utils.py`
  - `VisionTransformer.py`
  - `Demo.ipynb`
  - `requirements.txt`
  - `run.bat`
  - `TransUnet_15.pth` (can be obtained via: https://drive.google.com/file/d/1HptchVLvkEFXW1IPcPcJksFpS18HaQdx/view?usp=drive_link)

Please ensure the completeness of the above files and directory structure to successfully run the project.

## Usage Instructions

### 1. Install Dependencies

First, ensure that your Python version is 3.9.13, and install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

The `Demo.ipynb` provides a simple demonstration, where 12 images are used as training data, and 4 images are used as test data. The demo performs 5 random predictions based on the parameters from `TransUnet_15.pth`.

### 3. Train the Model

If you wish to train the model on the original dataset, modify the hyperparameters at the top of `trainer.py` as per your needs:

```python
TRAIN_SPLIT_RATIO = 0.3  # Training set ratio
TEST_SPLIT_RATIO = 0.1   # Test set ratio
```

After making the changes, run the `run.bat` script to perform the full training and prediction process.

## File Descriptions

- `dataset.py`: Script for dataset handling.
- `trainer.py`: Model training script, configurable for training and test set ratios.
- `TranUnet.py`: Implementation of the `TransUnet` model.
- `Unet.py`: Implementation of the `Unet` model.
- `utils.py`: Helper utility functions.
- `VisionTransformer.py`: Implementation of the Vision Transformer model.
- `Demo.ipynb`: Simple demo showing model prediction results.
- `requirements.txt`: Project dependency list.
- `run.bat`: Batch script for running the training and prediction process.
- `TransUnet_15.pth`: Pre-trained model parameter file.

## Contact Information

For any issues, please contact the project maintainer (z5446200@ad.unsw.edu.au)
