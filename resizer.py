## Bulk image resizer

# Adapted source code from original file located at:
# https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/resizer.py
# Thank you to Evan EdjeElectronics

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os
import sys

#dir_path = os.getcwd()
#dir_path = sys.argv[0]


def convert(dir_path):
   for filename in os.listdir(dir_path):
      # If the images are not .JPG images, change the line below to match the image type.
      if filename.endswith(".jpeg"):
         print("Filename = ", filename)
         image = cv2.imread(filename)
         resized = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
         cv2.imwrite(filename,resized)

if __name__ == "__main__":
   print(sys.argv[1])
   convert(sys.argv[1])
