import cv2

import torch
import torchvision.models as models

from .src.const import MODEL_CKP

class FKGTask:
    """
    Designed to handle and prepare data for classification tasks. 
    It provides methods to encode categorical labels, split data into training and validation sets, 
    and retrieve the encoded data, making it useful for machine learning tasks.

    Parameters:
    - face_side (str): The column name representing the face side in the dataset.
    - subtask (str): The column name representing the subtask or label in the dataset.
    - dataframe (pd.DataFrame): The Pandas DataFrame containing the encoded data.

    Attributes:
    - face_side (str): The column name representing the face side in the dataset.
    - subtask (str): The column name representing the subtask or label in the dataset.
    - dataframe (pd.DataFrame): The Pandas DataFrame containing the encoded data.
    - encoding_dict (dict): A dictionary that maps the original class labels to their encoded values.
    - X (pd.Series): The feature data (independent variable).
    - y (pd.Series): The target data (dependent variable).

    Methods:
    - __init__(self, face_side, subtask, data): Initializes an instance of the FKGTask class.
    - _get_dataframe(self, data, subtask): Encodes the categorical labels in the dataset and returns the encoded DataFrame and encoding dictionary.
    - _get_x_and_y(self): Extracts the feature and target data from the DataFrame.
    - get_train_test_split(self): Splits the data into training and validation sets and returns them as DataFrames.

    Example Usage:
    >>> fkg_task = FKGTask(face_side=`face_side`, subtask=`subtask`, data=my_data)
    >>> train_data, val_data = fkg_task.get_train_test_split()
    
    >>>  # Train a machine learning model using the train_data
    >>>  # Validate the model using the val_data

    Note:
    - This class is designed to work with Pandas DataFrames and assumes that the input data contains columns corresponding to the specified `face_side` and `subtask`.
    - It uses ordinal encoding to convert categorical labels into numerical values.
    - The class provides a convenient way to split the data into training and validation sets, maintaining the stratified distribution of the target variable.
    """

    def __init__(self, face_side, subtask, data):
        self.face_side = face_side
        self.subtask = subtask
        self.dataframe, self.encoding_dict = self._get_dataframe(data, self.subtask)
        self.X, self.y = self._get_x_and_y()

    def _get_dataframe(self, data, subtask):
        encoder = OrdinalEncoder()
        unique_values = data[subtask][subtask].unique().reshape(-1, 1)
        encoder.fit(unique_values)

        before_val = data[subtask][subtask]
        data[subtask][subtask] = encoder.transform(data[subtask][subtask].values.reshape(-1, 1))
        encoding_dict = {original_class: encoded_value for original_class, encoded_value in zip(data[subtask][subtask], before_val)}
        encoding_dict = {v: k for k, v in encoding_dict.items()}
        
        return data[subtask], encoding_dict

    def _get_x_and_y(self):
        X = self.dataframe[self.face_side]
        y = self.dataframe[self.subtask]

        return X, y

    def get_train_test_split(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y
            , stratify=self.y
            , test_size=0.2
            , random_state=42
        )

        return pd.concat([X_train, y_train], axis=1), pd.concat([X_val, y_val], axis=1)

def crop_face(image_path, target_width=300, target_height=400):
    """
    Crop and resize the detected face in an image.

    Parameters:
    - image_path (str): The path to the input image file.
    - target_width (int, optional): The desired width of the output face image (default is 300 pixels).
    - target_height (int, optional): The desired height of the output face image (default is 400 pixels).

    Returns:
    - original_image (numpy.ndarray): The cropped and resized face region as a NumPy array in Gray color format.
    - gray_image (numpy.ndarray): The cropped and resized face region as a NumPy array in Gray color format.

    Raises:
    - ValueError: If the image cannot be loaded from the given path or if no faces are detected in the image.

    This function takes an image file, detects the face in the image, and resizes it to match the specified target 
    width and height while maintaining the aspect ratio. The result is returned as a NumPy array in RGB and Gray color format.

    Example usage:
    >>> original_image, gray_iamge = crop_face("your_image.jpg")
    """

    image = cv2.imread(image_path)
    image_rb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Search for faces on the image.
    face_cascade = cv2.CascadeClassifier("src/models/pretrained/haarcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    x, y, w, h = faces[0]

    # Adjust height, so it will create a 3:4 (width:height) ratio.
    exp_ratio = 3 / 4
    h = int(w / exp_ratio)

    # Adjust y, as a pre-caution if it 
    # being cropped below the forehead.
    y -= int((image.shape[0] / target_height) * 35)
    
    # Add padding for the height, as a pre-caution
    # if it being cropped below the forehead.
    if y + h > image.shape[0]:
        minus_y = y + h - image.shape[0]
        y -= minus_y

    image_cropped = image_rb[y:y+h, x:x+w]
    image_cropped_resized = cv2.resize(image_cropped, (target_width, target_height))

    resized_rgb = image_cropped_resized
    resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)

    return resized_rgb, resized_gray

def load_model_checkpoint(model, ckp_path):
    """
    Load a PyTorch model from a checkpoint file.
    This function loads a pre-trained or saved PyTorch 
    model from a checkpoint file and returns the loaded model.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be loaded.
    - ckp_path (str): The path to the checkpoint file.

    Returns:
    - loaded_model (torch.nn.Module): The model loaded from the checkpoint.

    Example Usage:
    >>> # Load a pre-trained model from a checkpoint file
    >>> model = load_model_checkpoint(models.resnet18(pretrained=False), 'model_checkpoint.pth')

    Note:
    - Ensure that the model architecture in the checkpoint file matches the provided `model` argument.
    """
    temp_model = model
    checkpoint = torch.load(ckp_path)
    temp_model.load_state_dict(checkpoint)

    return temp_model

def get_models():
    """
    Get a dictionary of pre-trained models.
    This function loads and returns a dictionary of pre-trained PyTorch models, 
    including EfficientNet B0, ShuffleNet V2, and MobileNet V2.

    Returns:
    - model_dict (dict): A dictionary with model names as keys and the corresponding pre-trained models as values.

    Example Usage:
    >>> # Get a dictionary of pre-trained models
    >>> models_dict = get_models()
    >>> effnet_model = models_dict['effnet']
    >>> shufflenet_model = models_dict['shufflenet']
    >>> mobilenet_model = models_dict['mobilenet']

    Note:
    - Make sure to import the necessary PyTorch model modules from `torchvision.models`.
    """
    efficientnet_b0 = load_model_checkpoint(models.efficientnet_b0(pretrained=False), MODEL_CKP['effnet'])  
    shufflenet = load_model_checkpoint(models.shufflenet_v2_x1_0(pretrained=False), MODEL_CKP['shufflenet'])
    mobilenet_v2 = load_model_checkpoint(models.mobilenet_v2(pretrained=False), MODEL_CKP['mobilenet'])

    model_dict = {}
    model_dict['effnet'] = efficientnet_b0
    model_dict['shufflenet'] = shufflenet
    model_dict['mobilenet'] = mobilenet_v2

    return model_dict