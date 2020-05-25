import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--labels_source_folder", type=str, required=True)
parser.add_argument("--imgs_source_folder", type=str, required=True)
parser.add_argument("--extract_first", action='store_true', default=False, required=False)
parser.add_argument("--out_name", type=str, required=True)
args = parser.parse_args()

data = {'Image': [], 'Object Name': [],	'Object Index': [],	'bx': [], 'by': [],	'bw': [], 'bh': []}
dones = 0
for root, dirs, files in os.walk(args.labels_source_folder):
    for file in files:
        if file.endswith('.txt'):
            filename, file_extension = os.path.splitext(file)
            if file_extension == '.txt':
                img_fn = f'{filename}.jpg'
                img_path = os.path.join(args.imgs_source_folder, img_fn)
                with open(os.path.join(root, file)) as f:
                    for l in f:
                        row = list(map(float, l.strip().split()))
                        label, x, y, w, h = row[:5]
                        object_name = 'mask' if label == 0.0 else 'nomask'
                        data['Image'].append(img_fn)
                        data['Object Name'].append(object_name)
                        data['Object Index'].append(int(label))
                        data['bx'].append(x)
                        data['by'].append(y)
                        data['bw'].append(w)
                        data['bh'].append(h)

                        # hack for bug in crop-faces
                        if len(row) > 5 and not args.extract_first:
                            i = 5
                            while i < len(row):
                                label = 1.0
                                x, y, w, h = row[i: i + 4]
                                i += 4
                                object_name = 'mask' if label == 0.0 else 'nomask'
                                data['Image'].append(img_fn)
                                data['Object Name'].append(object_name)
                                data['Object Index'].append(int(label))
                                data['bx'].append(x)
                                data['by'].append(y)
                                data['bw'].append(w)
                                data['bh'].append(h)
                        elif args.extract_first:
                            break
                
                dones += 1
                if dones % 100 == 0:
                    print("done", dones)

pd.DataFrame(data, columns = ['Image', 'Object Name', 'Object Index', 'bx', 'by', 'bw', 'bh']).to_csv(args.out_name, index=False)

