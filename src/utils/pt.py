import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import os

from .functions import crop_face

class CustomDatasets(Dataset):
    def __init__(self, image_path):
        self.cropped, _ = crop_face(image_path)

class CustomDataset(Dataset):
    """
    Custom PyTorch dataset for working with image data and labels stored in a DataFrame.

    Parameters:
    - dataframe (pandas.DataFrame): The DataFrame containing image file paths and corresponding labels.
    - root (str): The root directory where the image files are located.
    - transform (callable, optional): A function/transform to apply to the images (e.g., data augmentation).

    Attributes:
    - dataframe (pandas.DataFrame): The input DataFrame containing image file paths and labels.
    - root (str): The root directory where image files are stored.
    - transform (callable, optional): A function/transform to be applied to the images.

    Methods:
    - __len__(): Returns the number of samples in the dataset.
    - __getitem__(idx): Returns the image and label for the specified index.

    This dataset class is designed to work with image data stored in a DataFrame, where each row contains
    a file path to an image and its corresponding label. It allows for data loading, cropping, and optional
    data transformation using PyTorch's data loading utilities. 

    Example usage:
    >>> dataset = CustomDataset(dataframe, root_dir, transform=transforms.Compose([transforms.Resize(256), transforms.ToTensor()]))
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for images, labels in dataloader:
    >>>     # Process the batch of images and labels
    """

    def __init__(self, dataframe, root, transform):
        self.dataframe = dataframe
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.dataframe.iloc[idx, 0])
        image, _ = crop_face(img_path)
        image = Image.fromarray(image)
        
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform: 
            image = self.transform(image)

        return image, label

class FKGLightningModuleV1(pl.LightningModule):
    """
    FKGLightningModuleV1 is a PyTorch Lightning Module for image classification tasks. It allows training and validation of
    models with different architectures, such as EfficientNet, MobileNet, and ShuffleNet for classification problems.

    Parameters:
    - num_classes (int): The number of classes for the classification task.
    - model_name (str): The name of the model architecture to use from the model_dict.
    - model_dict (dict): A dictionary containing pre-trained PyTorch models.

    Attributes:
    - num_classes (int): The number of classes for the classification task.
    - model_name (str): The name of the model architecture being used.
    - model_dict (dict): A dictionary containing pre-trained PyTorch models.
    - model (nn.Module): The neural network model loaded from model_dict.
    - criterion (nn.Module): The loss function, Cross-Entropy Loss, used for training.

    Methods:
    - _load_model(self): Loads the specified model architecture and adjusts it for the given task.
    - forward(self, x): Performs forward pass through the model.
    - configure_optimizers(self): Configures the optimizer for training.
    - training_step(self, batch, batch_idx): Defines a training step, including forward and loss calculation.
    - validation_step(self, batch, batch_idx): Defines a validation step, including forward and loss calculation.

    Example Usage:
    >>> # Create an instance of FKGLightningModuleV1
    >>> model = FKGLightningModuleV1(num_classes=10, model_name='effnet', model_dict=model_dict)

    >>> # Configure Lightning Trainer and train the model
    >>> trainer = pl.Trainer(gpus=1)
    >>> trainer.fit(model, train_dataloader, val_dataloader)

    Note:
    - This class is designed to work with PyTorch Lightning for efficient training and validation of image classification models.
    - It supports various model architectures, and you need to provide a `model_dict` containing pre-trained models.
    - Make sure to customize the model architecture for your specific classification task by modifying the `_load_model` method.
    """

    def __init__(self, num_classes, model_name, model_dict):
        super(FKGLightningModuleV1, self).__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.model_dict = model_dict
        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss()

    def _load_model(self):
        temp_model = self.model_dict[self.model_name]
        
        if self.model_name in ['effnet', 'mobilenet']:
            num_feat = temp_model.classifier[-1].in_features
            
        if self.model_name in ['shufflenet', 'resnet']:
            num_feat = temp_model.fc.in_features

        model = nn.Sequential(*list(temp_model.children())[:-1])
        model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
        model.add_module('flatten', nn.Flatten())
        model.add_module('fc', nn.Linear(num_feat, self.num_classes))
        
        return model
        
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        self.log('train_loss', loss, on_epoch=True)    
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)
        
        self.log('val_loss', val_loss, on_epoch=True)
        return val_loss