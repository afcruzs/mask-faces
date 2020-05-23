from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import os
import uuid
import matplotlib.image as mpimg
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=True)
parser.add_argument("--target_folder", type=str, required=True)
parser.add_argument("--workers", type=int, required=False, default=4)
args = parser.parse_args()
face_detector = MTCNN()

def crop_images(f_args):
    source_folder, target_folder, file_path = f_args
    try:
        img = mpimg.imread(file_path)
        if img.shape == 3 and img.shape[2] > 3:
            img = img[:,:,:3]

        bb = face_detector.detect_faces(img)
    except Exception as ex:
        print ("Error ", ex)
    
    if len(bb) > 0:
        print("Doing ", file_path)
    
    for data in bb:
        try:
            (x, y, x1, y1) = data['box']
            subimg = img[y: y + y1, x: x + x1]
            plt.imshow(subimg)
            name = uuid.uuid4().hex[:10]
            subimg_path = os.path.join(target_folder, f'{name}.jpg')
            im = Image.fromarray(subimg)
            im.save(subimg_path)
            print("saved", subimg_path)
        except:
            pass

if __name__ == '__main__':
       
    #crop_images(face_detector, args.source_folder, args.target_folder)

    f_args = [(args.source_folder, args.target_folder, os.path.join(root, file)) for root, dirs, files in os.walk(args.source_folder) for file in files]
    with Pool(args.workers) as p:
        p.map(crop_images, f_args)
            