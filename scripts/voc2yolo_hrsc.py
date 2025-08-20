# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 06:14:47 2025

@author: Gyhsh
"""

# -*- coding: utf-8 -*-
"""
HRSC2016 VOC -> YOLOv5/8 标签转换（稳健版）
- 支持：自动类别统计；默认合并为 1 类（ship）；或按提供/自动发现的完整类别集合映射
- 标注：优先用 <bndbox> 水平框；若缺失则从 <robndbox> 计算其外接水平框（AABB）
- 兼容：.bmp/.jpg/.jpeg/.png；list_file 是否带扩展名；空标注文件；进度与统计打印

用法示例（合并所有类别到 0:'ship'，推荐）：
python scripts/voc2yolo_hrsc.py \
  --voc_img_dir data/HRSC2016/AllImages \
  --voc_xml_dir data/HRSC2016/Annotations \
  --list_file data/HRSC2016/ImageSets/train.txt \
  --yolo_img_dir data/HRSC2016/images/train \
  --yolo_lbl_dir data/HRSC2016/labels/train

若保留原始类别（按出现顺序自动生成映射）：
python scripts/voc2yolo_hrsc.py ... --merge_all false
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse, shutil, math, sys
from collections import Counter, defaultdict

IMG_EXTS = [".bmp", ".jpg", ".jpeg", ".png"]

def find_image(img_root: Path, stem: str) -> Path | None:
    # 既兼容 list_file 只有 stem（如 100000001），也兼容带扩展名
    stem_clean = Path(stem).stem
    for ext in IMG_EXTS:
        p = img_root / f"{stem_clean}{ext}"
        if p.exists():
            return p
    return None

def read_list_ids(list_file: Path) -> list[str]:
    with open(list_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return ids

def parse_size(root: ET.Element) -> tuple[int,int]:
    sz = root.find("size")
    if sz is None:
        raise ValueError("XML缺少<size>标签")
    w = int(float(sz.findtext("width", "0")))
    h = int(float(sz.findtext("height", "0")))
    if w <= 0 or h <= 0:
        raise ValueError("XML的宽高无效")
    return w, h

def get_bndbox(obj: ET.Element) -> tuple[float,float,float,float] | None:
    """
    返回 (xmin, ymin, xmax, ymax) or None
    """
    bb = obj.find("bndbox")
    if bb is None: 
        return None
    try:
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def aabb_from_robndbox(obj: ET.Element) -> tuple[float,float,float,float] | None:
    """
    从 <robndbox> 计算其外接水平框 AABB。
    HRSC2016 的 angle 通常为弧度。
    """
    rb = obj.find("robndbox")
    if rb is None:
        return None
    try:
        cx = float(rb.findtext("cx"))
        cy = float(rb.findtext("cy"))
        w = float(rb.findtext("w"))
        h = float(rb.findtext("h"))
        angle = float(rb.findtext("angle"))
        if w <= 0 or h <= 0:
            return None
        dx, dy = w / 2.0, h / 2.0
        corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        xs, ys = [], []
        for u, v in corners:
            x = cx + u * cos_a - v * sin_a
            y = cy + u * sin_a + v * cos_a
            xs.append(x); ys.append(y)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax
    except Exception:
        return None

def clip_box(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(xmin, w - 1.0))
    ymin = max(0.0, min(ymin, h - 1.0))
    xmax = max(0.0, min(xmax, w - 1.0))
    ymax = max(0.0, min(ymax, h - 1.0))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax

def to_yolo(w, h, box):
    xmin, ymin, xmax, ymax = box
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    bw = (xmax - xmin)
    bh = (ymax - ymin)
    return xc / w, yc / h, bw / w, bh / h

def scan_classes(voc_xml_dir: Path, ids: list[str]) -> list[str]:
    seen = []
    seen_set = set()
    for sid in ids:
        xml = voc_xml_dir / f"{Path(sid).stem}.xml"
        if not xml.exists():
            continue
        try:
            root = ET.parse(xml).getroot()
            for obj in root.iter("object"):
                name = (obj.findtext("name") or "").strip()
                if name and name not in seen_set:
                    seen.append(name)
                    seen_set.add(name)
        except Exception:
            pass
    return seen

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Convert HRSC2016 VOC to YOLO")
    ap.add_argument("--voc_img_dir", required=True, help="HRSC2016/AllImages")
    ap.add_argument("--voc_xml_dir", required=True, help="HRSC2016/Annotations")
    ap.add_argument("--list_file", required=True, help="HRSC2016/ImageSets/{train|val|test}.txt")
    ap.add_argument("--yolo_img_dir", required=True)
    ap.add_argument("--yolo_lbl_dir", required=True)
    ap.add_argument("--merge_all", default="true", choices=["true","false"],
                    help="是否将所有类别合并为 0:'ship'（默认 true）")
    ap.add_argument("--classes", default="", help="保留的类别名列表，用逗号分隔；留空则自动扫描（当 merge_all=false 时生效）")
    ap.add_argument("--copy_images", default="true", choices=["true","false"], help="是否复制图片到目标目录")
    args = ap.parse_args()

    voc_img_dir = Path(args.voc_img_dir)
    voc_xml_dir = Path(args.voc_xml_dir)
    list_file = Path(args.list_file)
    yolo_img_dir = Path(args.yolo_img_dir)
    yolo_lbl_dir = Path(args.yolo_lbl_dir)
    merge_all = args.merge_all.lower() == "true"
    copy_images = args.copy_images.lower() == "true"

    ensure_dirs(yolo_img_dir, yolo_lbl_dir)

    ids = read_list_ids(list_file)
    if not ids:
        print(f"[ERR] list_file 为空：{list_file}")
        sys.exit(1)

    # 类别映射
    if merge_all:
        class_names = ["ship"]
        name_to_id = defaultdict(lambda: 0)  # 所有都映射到 0
        print("[INFO] 所有类别将合并为 0:'ship'")
    else:
        if args.classes.strip():
            class_names = [x.strip() for x in args.classes.split(",") if x.strip()]
        else:
            class_names = scan_classes(voc_xml_dir, ids)
        if not class_names:
            print("[ERR] 未发现任何类别，请检查 XML 或设置 --classes")
            sys.exit(1)
        name_to_id = {name: i for i, name in enumerate(class_names)}
        print("[INFO] 类别映射：")
        for i, n in enumerate(class_names):
            print(f"  {i}: {n}")

    # 统计
    stats_total_imgs = 0
    stats_copied_imgs = 0
    stats_total_objs = 0
    stats_written_labels = 0
    per_class = Counter()
    empty_labels = 0
    missed_imgs = 0
    missing_xml = 0

    for sid in ids:
        stem = Path(sid).stem
        xml_file = voc_xml_dir / f"{stem}.xml"
        if not xml_file.exists():
            missing_xml += 1
            continue

        img_file = find_image(voc_img_dir, stem)
        if img_file is None:
            missed_imgs += 1
            continue

        try:
            root = ET.parse(xml_file).getroot()
            W, H = parse_size(root)
        except Exception as e:
            print(f"[WARN] 解析尺寸失败，跳过: {xml_file} ({e})")
            continue

        labels = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                continue
            if merge_all:
                cls_id = 0
            else:
                if name not in name_to_id:
                    # 保留非白名单类别？这里选择跳过
                    continue
                cls_id = name_to_id[name]

            # 优先 <bndbox>，否则用 <robndbox> 的外接水平框
            box = get_bndbox(obj)
            if box is None:
                box = aabb_from_robndbox(obj)
            if box is None:
                continue

            box = clip_box(box, W, H)
            if box is None:
                continue

            x, y, w, h = to_yolo(W, H, box)
            # 过滤异常/退化框
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue

            labels.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            per_class[cls_id] += 1
            stats_total_objs += 1

        # 写标签
        ylbl = yolo_lbl_dir / f"{stem}.txt"
        with open(ylbl, "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
        stats_written_labels += 1
        if not labels:
            empty_labels += 1

        # 复制图片
        if copy_images:
            shutil.copy2(img_file, yolo_img_dir / img_file.name)
            stats_copied_imgs += 1

        stats_total_imgs += 1
        if stats_total_imgs % 200 == 0:
            print(f"[INFO] 进度: {stats_total_imgs}/{len(ids)}")

    # 汇总
    print("\n==== 转换完成 SUMMARY ====")
    print(f"子集图像数（list_file）：{len(ids)}")
    print(f"成功解析并写出标签：{stats_written_labels}")
    print(f"空标签文件数：{empty_labels}")
    print(f"缺失 XML 数：{missing_xml}")
    print(f"未找到图片数：{missed_imgs}")
    print(f"复制图片数：{stats_copied_imgs}")
    print(f"总目标数：{stats_total_objs}")
    if merge_all:
        print("类别：0:'ship'")
    else:
        print("类别直方图（cls_id: count）：")
        for cid, cnt in sorted(per_class.items()):
            print(f"  {cid}: {cnt}")
    print("==== END ====")

if __name__ == "__main__":
    main()