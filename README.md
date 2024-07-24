# Facial Keypoints Detection

This project implements a facial keypoints detection model using EfficientNet and PyTorch. The model is trained to predict the coordinates of key facial features in images.

## Project Structure

- `train.py`: Main script to train the model.
- `config.py`: Configuration file containing hyperparameters and other settings.
- `data_augs.py`: Data augmentation and transformation functions.
- `split_data.py`: Script to split the dataset into training and validation sets.
- `dataset.py`: Custom dataset class for loading the facial keypoints dataset.
- `utils.py`: Utility functions for training and evaluation.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/facial-keypoints-detection.git
   cd facial-keypoints-detection
   ```

## Create a Conda environment:

```bash
conda create --name keypoints_env python=3.11
conda activate keypoints_env
```

## Install dependencies:

```bash
conda install pytorch torchvision torchaudio -c pytorch
pip install efficientnet-pytorch pandas tqdm
```

## Dataset

Download the dataset from Kaggle:

### 1. Install Kaggle CLI:

```bash
pip install kaggle
```

### 2. API Token:

- Go to your Kaggle account and create an API token.
- Place the downloaded kaggle.json file in the directory ~/.kaggle/.

### 3. Download the dataset:

```bash
kaggle datasets download -d bravo03/facial-detection-keypoints
```

### 4. Extract the dataset:

```bash
unzip facial-detection-keypoints.zip -d facial-keypoints-detection/
```

The dataset should be organized as follows:

- facial-keypoints-detection/training.csv: Training data with facial keypoints.
- facial-keypoints-detection/test.csv: Test data without keypoints.

## Usage

### Split the training data:

The split_data.py script splits the training data into different sets for training and validation.

```python
python split_data.py
```

### Train the model:

Run the train.py script to start training the model.

```python
python train.py
```

### Functions

- train.py
- train_one_epoch(loader, model, optimizer, loss_fn, device): Trains the model for one epoch.
- main(): Main function to setup data loaders, model, loss function, optimizer, and start the training process.
- split_data.py
- manual_split_training_data(): Splits the training data into train_4, val_4, train_15, and val_15 sets and saves them as CSV files.

### Configuration

The config.py file contains configuration settings such as batch size, learning rate, number of epochs, and file paths. Modify this file to adjust the training parameters.

## Acknowledgements

This project uses the EfficientNet implementation from efficientnet-pytorch.
The dataset used for training is part of the Facial Keypoints Detection competition on Kaggle.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or additions.
