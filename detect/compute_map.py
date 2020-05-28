from mapcalc import calculate_map as calculate_map
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
import imagesize
import cv2
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("map.log"),
        logging.StreamHandler()
    ]
)

def _compute_map(predictions, true_boxes, iou_threshold):
    total = 0.0
    n = 0

    for preds, trues in zip(predictions, true_boxes):

        pred_detections = {'labels': [], 'boxes': [], 'scores': []}
        true_detections = {'labels': [], 'boxes': [], 'scores': []}
        
        for pred in preds:
            x_pred, y_pred, w_pred, h_pred, c_pred, label_pred = pred
            x1_pred = x_pred + w_pred
            y1_pred = y_pred + h_pred
            pred_detections['labels'].append(label_pred)
            pred_detections['boxes'].append((x_pred, y_pred, x1_pred, y1_pred))
            pred_detections['scores'].append(c_pred)

        for true in trues:
            x, y, w, h, c, l = true
            x1 = x + w
            y1 = y + h        
            true_detections['labels'].append(l) 
            true_detections['boxes'].append((x, y, x1, y1))
            true_detections['scores'].append(c)

        total += calculate_map(true_detections, pred_detections, iou_threshold)
        n += 1

    total /= n
    return total

def compute_map_from_dataset(data_path, masks_detector, iou_threshold, max_imgs=float('inf'), verbose=0):

    predictions = []
    true_predictions = []
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        for idx, filename in enumerate(filenames):
            name, file_extension = os.path.splitext(filename)
            if file_extension == '.jpg':
                img_path = os.path.join(dirpath, filename)
                
                if verbose >= 1 and idx % 100 == 0:
                    logging.info(str(idx) + " " + img_path)
                
                img = image.load_img(img_path)
                results, faces_detector_output = masks_detector.detect(np.array(img), verbose=0)

                model_predictions = []
                for label, detection in zip(results, faces_detector_output):
                    model_predictions.append(tuple(detection['box']) + (detection['confidence'], label))


                ground_truth = []
                img_width, img_height = imagesize.get(os.path.join(dirpath, filename))
                with open(os.path.join(dirpath, f'{name}.txt')) as f:
                    for line in f:
                        object_class, x, y, width, height, = map(float, line.strip().split())
                        object_class = int(object_class)
                        confidence = 1.0 # true pred
                        x *= img_width
                        y *= img_height
                        width *= img_width
                        height *= img_height
                        ground_truth.append((x, y, width, height, confidence, object_class))


                predictions.append(model_predictions)
                true_predictions.append(ground_truth)
                if len(predictions) >= max_imgs:
                    break

    return _compute_map(predictions, true_predictions, iou_threshold)