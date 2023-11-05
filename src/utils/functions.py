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
    if image is None:
        raise ValueError("Unable to load the image from the given path.")

    image = cv2.resize(image, (target_width, target_height))
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the face cascade classifier and detect faces on the image.
    face_cascade = cv2.CascadeClassifier("src/models/pretrained/haarcascade_frontalface.xml")
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    # Images that will be processed in this project
    # will be ensured that it only contains 1 face.
    x, y, w, h = faces[0]
    padding = int(max(0.15 * (x + w), 0.15 * (y + h))) # Add padding to create space.

    width_scale, height_scale = target_width / w, target_height / h
    scale = min(width_scale, height_scale)

    original_image = cv2.resize(original_image[y-padding:y+h+padding, x:x+w], None, fx=scale, fy=scale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return original_image, gray_image
