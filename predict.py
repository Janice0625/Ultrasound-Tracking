import sys
sys.path.append("C:\\Users\\Janice\\yolov10")

import torch
import cv2
import os
from pathlib import Path
from yolov10.models.experimental import attempt_load
from yolov10.utils.general import non_max_suppression, scale_coords
from yolov10.utils.datasets import letterbox
from yolov10.utils.plots import plot_one_box

# ==== 手動設定參數 ====
weights = 'C:\\Users\\Janice\\yolov10\\runs\\train\\train5\\weights\\best.pt'  # 訓練完成的 YOLOv10 模型
source = 'C:\\Users\\Janice\\yolov10\\runs\\detect\\9_stitching_170.png'     # 要推論的單張圖片
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = 'C:\\Users\\Janice\\yolov10\\runs\\detect'  # 指定輸出資料夾

# ==== 載入模型 ====
model = attempt_load(weights, map_location=device)
model.eval()

# ==== 載入並預處理圖片 ====
img0 = cv2.imread(source)
assert img0 is not None, f"❌ 無法讀取圖片: {source}"
img = letterbox(img0, new_shape=imgsz)[0]
img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
img = torch.from_numpy(img).to(device).float() / 255.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# ==== 推論 ====
with torch.no_grad():
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

# ==== 繪製與儲存結果 ====
os.makedirs(save_dir, exist_ok=True)
img_name = Path(source).name
out_path = os.path.join(save_dir, img_name)

for i, det in enumerate(pred):
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{int(cls)} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=(255, 0, 0), line_thickness=2)

cv2.imwrite(out_path, img0)
print(f"\n✅ 推論完成，結果儲存於：{out_path}")
