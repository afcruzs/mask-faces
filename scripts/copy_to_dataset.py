import os
import random
import shutil
import argparse
import pickle
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="data path", required=True)
parser.add_argument("--number", type=int, help="Amount of data to copy", required=True)
parser.add_argument("--output_path", type=str, help="output data path", required=True)
args = parser.parse_args()

random.seed(42)

all_file_names = [os.path.join(r, f) for r, d, files in os.walk(args.input_path) for f in files]
random.shuffle(all_file_names)

print(len(all_file_names))

n = 0
for fn in all_file_names:
    try:
        nfn = uuid.uuid4().hex[:20]
        nfn += '.jpg' # are all images jpg? 
        shutil.copy2(fn, args.output_path)
    except:
        pass
    if n >= args.number:
        break

    if n % 100 == 0:
        print(n, "copied")
    
    n += 1
