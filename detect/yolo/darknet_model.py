import sys
from .darknet import load_net_custom, load_meta, make_image, network_width, network_height, copy_image_from_bytes, detect_image
import os
import cv2

netMain = None
metaMain = None
altNames = None

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def convert_points(x, y, w, h):
    return list(map(int, [x, y, w, h]))
    # shameless copy from video example

    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    return [xmin, ymin, xmax - xmin, ymax - ymin]

class DarknetModel(object):

    def __init__(self, configPath, weightPath, metaPath):
        global metaMain, netMain, altNames
        
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")
        if netMain is None:
            netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = load_meta(metaPath.encode("ascii"))
        if altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("/n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.darknet_image = make_image(network_width(netMain), network_height(netMain), 3)
    
    def detect(self, image, verbose=0):
        global metaMain, netMain, altNames

        frame_resized = cv2.resize(image,
                                (network_width(netMain),
                                network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)

        copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = detect_image(netMain, metaMain, self.darknet_image, thresh=0.25)
        
        # ((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)
        # res.append((meta.names[i], dets[j].prob[i], (b.x - b.w / 2, b.y - b.h /2, b.w, b.h)))?
        detector_output = [{'confidence': prob, 'box': convert_points(x, y, w, h)} for tag, prob, (x, y, w, h) in detections]
        results = [1 if x == b'notmask' else 0 for x, _, _ in detections]

        for x, _, _ in detections:
            if x != b'notmask' and x != b'mask':
                print(x)
                raise "this is wrong!"

        return results, detector_output


if __name__ == "__main__":
    
    print(sys.argv[1])
    
    configPath = "../../../cfg/yolov3-tiny-masks-small.cfg"
    weightPath = "../../../backup-20000/yolov3-tiny-masks-small_last.weights"
    metaPath = "faces-20000.data"
    
    model = DarknetModel(configPath, weightPath, metaPath)
    img = cv2.imread(sys.argv[1]) 
    print(model.detect(img))
    print(model.detect(img))
    print(model.detect(img))
    