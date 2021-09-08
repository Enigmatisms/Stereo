"""
    Ground truth readin & saving tool for
    Middlebury dataset
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack

# PFM reader is from: https://stackoverflow.com/questions/37073108/how-to-read-pfm-files-provided-with-middlebury-dataset
def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)
        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
    return img, height, width

def generateUsableGT():
    for root, dirs, _ in os.walk("../data/MiddEval3/trainingQ/"):
        for _dir in dirs:
            path_prefix = os.path.join(root, _dir)
            gt_path = "%s/%s"%(path_prefix, "disp0GT.pfm")
            calib_path = "%s/%s"%(path_prefix, "calib.txt")
            ndisp = 0
            print(gt_path)
            with open(calib_path, 'r') as calib:
                while True:
                    line = calib.readline()
                    if not line: break
                    name, val = line.split("=")
                    if name == "ndisp":
                        ndisp = int(val)
                        break
            gt, height, width = read_pfm(gt_path)
            gt = np.reshape(gt, (height, width))
            gt = np.fliplr([gt])[0]
            gt[gt == float('inf')] = 0.0
            data = np.array([width, height, ndisp])
            gt = np.concatenate([data, gt.ravel()])
            save_path = "%s/%s"%(path_prefix, "gt_disp.bin")
            gt.tofile(save_path)

def testSavedFiles():
    for root, dirs, _ in os.walk("../data/MiddEval3/trainingQ/"):
        for _dir in dirs:
            path_prefix = os.path.join(root, _dir)
            gt_path = "%s/%s"%(path_prefix, "gt_disp.bin")
            data = np.fromfile(gt_path)
            w, h, _ = data[:3].astype(int)
            img = data[3:].reshape(h, w)
            plt.imshow(img)
            plt.colorbar()
            plt.show()
            break

if __name__ == "__main__":
    # testSavedFiles()
    generateUsableGT()
            