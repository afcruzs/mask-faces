import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, help="Folder where the data is", required=True)
parser.add_argument('--relative_labels', type=str, help='path to csv file with labels', required=True)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    df = pd.read_csv(args.relative_labels)
    imgs = set(df.Image.to_list())
    ctr = 0
    for root, dirs, files in os.walk(args.dataset_folder):
        for file in files:
            if file not in imgs:
                os.remove(os.path.join(root, file))
                ctr += 1
                if ctr % 100 == 0:
                    print(ctr, "Deleted")

    
    print(len(imgs), ctr)