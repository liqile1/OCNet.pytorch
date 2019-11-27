import os
import cv2
import numpy as np

if __name__ == "__main__":
    mis_ok = 0
    mis_ng = 0
    for idx in range(1463, 2352):
        label = cv2.imread('dataset/leadbang/label/' + str(idx) + '.bmp', 0)
        result = cv2.imread('dataset/leadbang/test_result/' + str(idx) + '.bmp', 0)
        label_defect = len(np.where(label < 100)[0])
        result_defect = len(np.where(result < 100)[0])
        if label > 0 and result_defect == 0:
            mis_ok += 1
        if label == 0 and result_defect > 0:
            mis_ng += 1
    print('mis ok: ', mis_ok)
    print('mis ng: ', mis_ng)