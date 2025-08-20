import cv2, glob, random

img_files = glob.glob("data/SSDD/images/train/*.jpg")
for img_path in random.sample(img_files, 5):
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
            x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow("check", img)
    cv2.waitKey(0)