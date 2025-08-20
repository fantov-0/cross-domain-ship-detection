import cv2, pathlib

img_path = pathlib.Path('data/HRSC2016/images/train/100000001.bmp')
lbl_path = pathlib.Path('data/HRSC2016/labels/train/100000001.txt')

img = cv2.imread(str(img_path))
h, w = img.shape[:2]

with open(lbl_path) as f:
    for line in f:
        cls, xc, yc, ww, hh = map(float, line.split())
        x1, y1 = int((xc - ww/2)*w), int((yc - hh/2)*h)
        x2, y2 = int((xc + ww/2)*w), int((yc + hh/2)*h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow('check', img)
cv2.waitKey(0)