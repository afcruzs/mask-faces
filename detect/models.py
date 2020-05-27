import numpy as np
import cv2
from timeit import default_timer as timer

'''
Detects masks in two phases:
1. Runs a Multi Task Cascade Convolutional network to detect faces
2. Runs a small convnet for classifying each face into mask or no mask
'''
class DLPipelineMaskDetector(object):
    
    def __init__(self, input_img_size, model, detector, model_preprocessing_fn=None, object_detector_confidence=0.5):
        self.input_img_size = input_img_size
        self.model = model
        self.detector = detector
        self.model_preprocessing_fn = model_preprocessing_fn
        self.object_detector_confidence = object_detector_confidence

    def _extract_img(self, img, data):
        (x, y, w, h) = data['box']
        x, y = max(0, x), max(0, y)
        x1 = x + w
        y1 = y + h
        sub = img[y: y1, x: x1, :] if self.model_preprocessing_fn is None else self.model_preprocessing_fn(img[y: y1, x: x1, :])
        return sub
    
    # TODO can image be a batch?
    def predict(self, image, verbose=0):
        debug = verbose >= 1

        if debug:
            print("Detecting imgs")

        start = timer()
        detector_output = [data for data in self.detector.detect_faces(image) if data['confidence'] >= self.object_detector_confidence]
        end = timer()
        if debug:
            print("Faces Detection lasted %.4f seconds" % float(end - start))
        
        if debug:
            print("Building batch ")
        
        start = timer()
        batch = np.array([cv2.resize(self._extract_img(image, data), self.input_img_size) for data in detector_output])
        end = timer()
        if debug:
            print("Cropping images lasted %.4f seconds" % float(end - start))

        if len(batch) == 0:
            return [], []

        if debug:
            print("Predicting")

        start = timer()
        results = self.model.predict(batch)
        results = [int(np.round(p[0])) for p in results]
        end = timer()
        if debug:
            print("Batched binary classification lasted %.4f seconds" % float(end - start))
        
        if debug:
            print("Done")
        
        return results, detector_output

    '''
    Detects image loaded previously.
    Returns (results, detector_output)

    results => List of integers where 0 means mask and 1 means no mask.
    detector_output => List of an object like this:
    {
        'confidence': -> Confidence on the face prediction
        'box' -> [x, y, w, h]
    }
    '''
    def detect(self, image, verbose=0):
        img = np.array(image)

        if len(img.shape) > 2 and img.shape[2] == 4:
            #convert the image from RGBA2RGB (png weird cases)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return self.predict(img, verbose=verbose)