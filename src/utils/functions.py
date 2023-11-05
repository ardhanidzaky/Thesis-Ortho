import cv2

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