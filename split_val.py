import os
import random
import shutil
from pathlib import Path

# ======== 参数设置 ========
base_dir = Path(r"C:\Users\Gyhsh\DGShip\data\SSDD")
split_ratio = 0.1  # 验证集占训练集比例，例如 0.1 表示 10%
random.seed(42)  # 保证可复现
# ========================

train_img_dir = base_dir / "images" / "train"
train_lbl_dir = base_dir / "labels" / "train"
val_img_dir   = base_dir / "images" / "val"
val_lbl_dir   = base_dir / "labels" / "val"

# 创建 val 文件夹
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# 列出所有训练图片
images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
print(f"[INFO] 训练集图片总数: {len(images)}")

# 随机抽取部分作为 val
val_count = int(len(images) * split_ratio)
val_images = random.sample(images, val_count)

for img_path in val_images:
    lbl_path = train_lbl_dir / (img_path.stem + ".txt")

    # 移动图片
    shutil.move(str(img_path), str(val_img_dir / img_path.name))

    # 移动对应标签（如果有）
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

print(f"[DONE] 已切分 {val_count} 张图片到验证集: {val_img_dir}")