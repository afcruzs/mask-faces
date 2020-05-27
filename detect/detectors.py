import sys
sys.path.append('..')

from .models import DLPipelineMaskDetector
from train.cnn import simple_cnn
from mtcnn import MTCNN

'''
Creates model with simple convnet binary classifier and mtcnn face detector
'''
def mtcnn_simple_cnn(network_pretrained_weights_path, object_detector_confidence=0.5):
    INPUT_IMG_SIZE = (160, 160)
    network = simple_cnn(INPUT_IMG_SIZE, pretrained_weights=network_pretrained_weights_path)
    face_detector = MTCNN()
    prepro = lambda img: img / 255.0 # Should be equivalent to test_datagen = ImageDataGenerator(rescale=1./255)?
    return DLPipelineMaskDetector(INPUT_IMG_SIZE, network, face_detector, object_detector_confidence=object_detector_confidence)