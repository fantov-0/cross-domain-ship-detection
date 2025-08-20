# -*- coding: utf-8 -*-
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse, shutil

def convert_box(size, box):
    dw, dh = 1.0/size[0], 1.0/size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x*dw, y*dh, w*dw, h*dh

def voc_to_yolo(voc_img_dir, voc_xml_dir, list_file, yolo_img_dir, yolo_lbl_dir, classes):
    Path(yolo_img_dir).mkdir(parents=True, exist_ok=True)
    Path(yolo_lbl_dir).mkdir(parents=True, exist_ok=True)
    with open(list_file) as f:
        ids = [x.strip() for x in f.readlines()]
    for img_id in ids:
        xml_file = Path(voc_xml_dir)/f"{img_id}.xml"
        img_file = Path(voc_img_dir)/f"{img_id}.bmp"
        if not xml_file.exists() or not img_file.exists():
            continue
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        size = root.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)
        lbls = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = [float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text)]
            x,y,ww,hh = convert_box((w,h), b)
            lbls.append(f"{cls_id} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f}")
        shutil.copy2(img_file, Path(yolo_img_dir)/img_file.name)
        with open(Path(yolo_lbl_dir)/f"{img_id}.txt","w") as f:
            f.write("\n".join(lbls))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_img_dir", required=True)
    ap.add_argument("--voc_xml_dir", required=True)
    ap.add_argument("--list_file", required=True)
    ap.add_argument("--yolo_img_dir", required=True)
    ap.add_argument("--yolo_lbl_dir", required=True)
    ap.add_argument("--classes", default="ship,Ship")
    args = ap.parse_args()
    classes = args.classes.split(",")
    voc_to_yolo(args.voc_img_dir, args.voc_xml_dir, args.list_file,
                args.yolo_img_dir, args.yolo_lbl_dir, classes)