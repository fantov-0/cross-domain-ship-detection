from ultralytics import YOLO
import matplotlib.pyplot as plt

# 模型权重路径
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# 在混合验证集上评估
metrics = model.val(data="data/hrsc_ssdd.yaml", split="val")

# 生成 PR 曲线
metrics.plot_pr_curve()  # 会保存到 runs/detect/val/pr_curve.png

# 生成混淆矩阵
metrics.plot_confusion_matrix()

# 生成 mAP 曲线
metrics.plot_results()

print("评估完成，图表已保存到验证结果目录")