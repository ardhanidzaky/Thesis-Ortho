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