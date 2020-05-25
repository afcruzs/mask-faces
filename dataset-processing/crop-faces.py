from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import os
import uuid
import matplotlib.image as mpimg
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=True)
parser.add_argument("--target_folder", type=str, required=True)
parser.add_argument("--mode", type=str, required=True, choices=('crop', 'bbox'))
args = parser.parse_args()
face_detector = MTCNN()

def crop_images(source_folder, target_folder, file_path):
    try:
        img = mpimg.imread(file_path)
        if img.shape == 3 and img.shape[2] > 3:
            img = img[:,:,:3]
        
        img_height, img_width = 1.0 * img.shape[0], 1.0 * img.shape[1]

        bb = face_detector.detect_faces(img)
    except Exception as ex:
        print ("Error ", ex)
        return
    
    if len(bb) > 0:
        print("Doing ", file_path)
    
    for data in bb:
        try:
            (x, y, width, height) = data['box']
            
            if args.mode == 'crop':
                subimg = img[y: y + height, x: x + width]
                plt.imshow(subimg)
                name = uuid.uuid4().hex[:10]
                subimg_path = os.path.join(target_folder, name + '.jpg')
                im = Image.fromarray(subimg)
                im.save(subimg_path)
                print("saved", subimg_path)
            elif args.mode == 'bbox':
                img_fn = os.path.basename(file_path)
                img_name = os.path.splitext(img_fn)[0]
                label_fn = f'{img_name}.txt'
                label_fp = os.path.join(target_folder, label_fn)

                with open(label_fp, 'a') as file:
                    # File format: <object-class> <x> <y> <width> <height>
                    # Where x, y, width, and height are relative to the image's width and height.
                    file.write(f'1 {x / img_width} {y / img_height} {width / img_width} {height / img_height}')
                
                print('saved', label_fp)
        except Exception as ex:
            pass

if __name__ == '__main__':   
    for root, dirs, files in os.walk(args.source_folder):
        for file in files:
            crop_images(args.source_folder, args.target_folder, os.path.join(root, file))
            