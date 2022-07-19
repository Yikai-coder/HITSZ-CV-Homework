import numpy as np
import os 
import cv2

def load_data(path):
    imgs = []
    labels = []
    os.chdir(path)
    for root, dir, files in os.walk("."):
        label = int(root.split("Person")[1])
        for file in files:
            img = cv2.readim(os.getcwd()+file, cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
            labels.append(label)
    print("1")
    print(len(imgs))
    print(len(labels))
    return imgs, labels

if __name__ == "__MAIN__":
    load_data(".\\data")