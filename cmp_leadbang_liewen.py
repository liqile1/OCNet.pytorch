import os
import cv2
import numpy as np

if __name__ == "__main__":
    mis_ok = 0
    mis_ng = 0
    ng = 0
    ok = 0
    for idx in range(1129, 1589):
        label = cv2.imread('dataset/leadbang_liewen/label/' + str(idx) + '.bmp', 0)
        result = cv2.imread('dataset/leadbang_liewen/test_result/' + str(idx) + '.bmp', 0)
        label_defect = len(np.where(label < 100)[0])
        result_defect = len(np.where(result < 100)[0])
        if label_defect > 0 and result_defect == 0:
            mis_ok += 1
        if label_defect == 0 and result_defect > 0:
            mis_ng += 1

        if label_defect > 0:
           ng += 1
        else:
           ok += 1
    print('mis ok: ', mis_ok, '/', ng)
    print('mis ng: ', mis_ng, '/', ok)
