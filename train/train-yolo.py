import sys
sys.path.append('od')

from Main.trainer import Trainer
from Config.augmentation_options import preset_1
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classes_file", type=str, help="classes file", required=True)
parser.add_argument('--relative_labels', type=str, help='path to csv file with labels', required=True)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    DATASET_NAME = 'mask-faces'

    tr = Trainer(
            input_shape=(160, 160, 3),
            classes_file=args.classes_file,
            image_width=640,  # The original image width
            image_height=480   # The original image height
    )

    dataset_conf = {
                'relative_labels': args.relative_labels,
                'dataset_name': DATASET_NAME,
                'test_size': 0.1,
                'sequences': preset_1,  # check Config > augmentation_options.py
                'augmentation': True,
    }

    anchors_conf = {
                'anchors_no': 9,
                'relative_labels':  args.relative_labels
    }

    tr.train(epochs=100, 
            batch_size=8, 
            learning_rate=1e-3, 
            dataset_name=DATASET_NAME, 
            merge_evaluation=False,
            min_overlaps=0.5,
            new_dataset_conf=dataset_conf,  # check step 5
            new_anchors_conf=anchors_conf  # check step 6
            #  weights='/path/to/weights'  # If you're using DarkNet weights or resuming training
            )