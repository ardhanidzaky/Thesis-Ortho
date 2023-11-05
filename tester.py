from src.utils.functions import crop_face
from src.utils.pt import CustomDatasets

def crop_face_test():
    image, gray = crop_face('data/image/test_img.jpg')
    
    expected_shape = (400, 300, 3)
    assert image.shape == expected_shape

def custom_dataset_test():
    cd = CustomDatasets('data/image/test_img.jpg')

    expected_shape = (400, 300, 3)
    assert cd.cropped == expected_shape

def test_all():
    crop_face_test()

if __name__ == '__main__':
    test_all()