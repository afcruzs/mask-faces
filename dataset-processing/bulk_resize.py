from PIL import Image
import argparse
from multiprocessing import Pool
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder", type=str, required=True)
parser.add_argument("--target_folder", type=str, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--workers", type=int, required=False, default=1)
args = parser.parse_args()

def resize_images(f_args):
    try:
        source_folder, target_folder, file_name, target_width, target_height = f_args
        img = Image.open(os.path.join(source_folder, file_name))
        new_img = img.resize((target_width, target_height)).convert('RGB')
        new_img.save(os.path.join(target_folder, file_name))
    except Exception as ex:
        print(ex)
        pass


if __name__ == '__main__':
    f_args = [(args.source_folder, args.target_folder, file, args.width, args.height) for root, dirs, files in os.walk(args.source_folder) for file in files]

    #for i in range(len(f_args)):
    #    resize_images(f_args[i])
    with Pool(args.workers) as p:
        p.map(resize_images, f_args)
            