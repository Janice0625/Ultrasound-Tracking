import os
import cv2
import numpy as np
from glob import glob
from collections import deque

# 建立不重複的輸出資料夾
def get_next_output_folder(base_path):
    i = 1
    while True:
        folder_name = f"track_output_{i}"
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        i += 1

# 設定輸入與輸出資料夾路徑
input_dir = "C://Users//user//Downloads//yolotrack//data//S0000001_predict"
base_output_dir = "C://Users//user//Downloads//yolotrack"
output_dir = get_next_output_folder(base_output_dir)

image_paths = sorted(glob(os.path.join(input_dir, "after_*.jpg")),
                     key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))

# 定義紀錄用 deque 和失幀計數器
recent_annotations = deque(maxlen=10)  # 儲存前 10 幀的標註資料
empty_frame_count = 0  # 連續無框幀數

# 將 YOLO txt 檔轉為中心點與框座標
def yolo_to_centers(txt_path, img_shape):
    h, w = img_shape[:2]
    centers = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            _, x, y, bw, bh = map(float, parts)
            cx = int(x * w)
            cy = int(y * h)
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            bw = int(bw * w)
            bh = int(bh * h)
            centers.append((cx, cy, (x1, y1, bw, bh)))
    centers.sort(key=lambda p: p[0])  # 按 X 座標排序
    return centers

# 自動補標缺失節點
def auto_label_missing_bones(annotations, centers, used_indices, max_move_threshold=50):
    all_labels = [
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
        "L1", "L2", "L3", "L4", "L5",
        "S1", "S2", "S3", "S4", "S5"
    ]
    label_to_index = {label: i for i, label in enumerate(all_labels)}
    index_to_label = {i: label for i, label in enumerate(all_labels)}
    label_positions = {label: (cx, cy) for label, (cx, cy) in annotations.items()}

    for i, label in enumerate(all_labels):
        if label in label_positions:
            continue  # 已存在，不補

        prev_idx = i - 1
        next_idx = i + 1
        if prev_idx < 0 or next_idx >= len(all_labels):
            continue  # 沒有左右相鄰節點

        prev_label = all_labels[prev_idx]
        next_label = all_labels[next_idx]

        if prev_label in label_positions and next_label in label_positions:
            x1, y1 = label_positions[prev_label]
            x2, y2 = label_positions[next_label]
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # 在中心點附近找最接近的未使用框
            best_match = None
            min_dist = float('inf')
            for j, (cx, cy, box) in enumerate(centers):
                if j in used_indices:
                    continue
                dist = np.sqrt((cx - mid_x) ** 2 + (cy - mid_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = (j, cx, cy)

            if best_match and min_dist < max_move_threshold:
                j, cx, cy = best_match
                annotations[label] = (cx, cy)
                used_indices.add(j)

    # 基於現有補中間節點邏輯，再補單邊邏輯
    for i, label in enumerate(all_labels):
        if label in annotations:
            continue

        # 嘗試用前一節補
        if i > 0 and all_labels[i - 1] in annotations:
            prev_label = all_labels[i - 1]
            x1, y1 = annotations[prev_label]
            for j, (cx, cy, box) in enumerate(centers):
                if j in used_indices:
                    continue
                if cx > x1 and abs(cx - x1) < max_move_threshold:
                    annotations[label] = (cx, cy)
                    used_indices.add(j)
                    break

        # 嘗試用後一節補
        elif i < len(all_labels) - 1 and all_labels[i + 1] in annotations:
            next_label = all_labels[i + 1]
            x2, y2 = annotations[next_label]
            for j, (cx, cy, box) in enumerate(centers):
                if j in used_indices:
                    continue
                if cx < x2 and abs(cx - x2) < max_move_threshold:
                    annotations[label] = (cx, cy)
                    used_indices.add(j)
                    break

    return annotations

def annotate_bones(image_path, txt_path, output_path, prev_annotations_list=None, max_move_threshold=40, shift_threshold=60):
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        centers = yolo_to_centers(txt_path, img.shape)
    except Exception as e:
        print(f"⚠️ 無法讀取 {txt_path}：{e}")
        return prev_annotations

    annotations = {}
    used_indices = set()

    # 整合前十幀標註進行追蹤
    if prev_annotations_list:
        for prev_annotations in prev_annotations_list:
            for label, (prev_x, prev_y) in prev_annotations.items():
                if label in annotations:
                    continue  # 已標註過
                best_match = None
                min_dist = float('inf')
                for i, (cx, cy, box) in enumerate(centers):
                    if i in used_indices:
                        continue
                    dist = np.sqrt((prev_x - cx) ** 2 + (prev_y - cy) ** 2)
                    if dist < min_dist and dist < shift_threshold:
                        min_dist = dist
                        best_match = (i, cx, cy, box)
                if best_match:
                    i, cx, cy, box = best_match
                    annotations[label] = (cx, cy)
                    used_indices.add(i)

        # 補 S2~S5
        for current, next_label in zip(["S1", "S2", "S3", "S4"], ["S2", "S3", "S4", "S5"]):
            if current in annotations and next_label not in annotations:
                x_cur, _ = annotations[current]
                for i, (cx, cy, box) in enumerate(centers):
                    if i in used_indices:
                        continue
                    if cx > x_cur and abs(cx - x_cur) < 100:
                        annotations[next_label] = (cx, cy)
                        used_indices.add(i)
                        break

        # 補 L4~L1
        for current, prev_label in zip(["L5", "L4", "L3", "L2"], ["L4", "L3", "L2", "L1"]):
            if current in annotations and prev_label not in annotations:
                x_cur, _ = annotations[current]
                for i, (cx, cy, box) in enumerate(centers):
                    if i in used_indices:
                        continue
                    if cx < x_cur and abs(cx - x_cur) < 100:
                        annotations[prev_label] = (cx, cy)
                        used_indices.add(i)
                        break

        # 若 S1 消失但 L5 存在 → 補 S1
        if "S1" not in annotations and "L5" in annotations:
            x_l5, _ = annotations["L5"]
            for i, (cx, cy, box) in enumerate(centers):
                if i in used_indices:
                    continue
                if x_l5 < cx < x_l5 + 100:
                    annotations["S1"] = (cx, cy)
                    used_indices.add(i)
                    break

        # 若 L5 消失但 S1 存在 → 補 L5
        if "L5" not in annotations and "S1" in annotations:
            x_s1, _ = annotations["S1"]
            for i, (cx, cy, box) in enumerate(centers):
                if i in used_indices:
                    continue
                if x_s1 - 100 < cx < x_s1:
                    annotations["L5"] = (cx, cy)
                    used_indices.add(i)
                    break

    else:
        # 第一幀：使用斜率找 S1 與 L5
        if len(centers) < 2:
            print(f"⚠️ 節點不足，跳過 {image_path}")
            return None

        slopes = []
        for i in range(len(centers) - 1):
            (x1, y1, _), (x2, y2, _) = centers[i], centers[i + 1]
            dx = x2 - x1 if (x2 - x1) != 0 else 1e-5
            slope = (y2 - y1) / dx
            slopes.append(slope)

        max_change, index_of_max = 0, 0
        for i in range(1, len(slopes)):
            change = abs(slopes[i] - slopes[i - 1])
            if change > max_change:
                max_change = change
                index_of_max = i

        slope_threshold = 0.3
        has_s1 = max_change > slope_threshold
        s1_index = index_of_max + 1 if has_s1 else 0
        s1_index = min(s1_index, len(centers) - 1)

        annotations["S1"] = centers[s1_index][:2]
        used_indices.add(s1_index)

        if s1_index - 1 >= 0:
            annotations["L5"] = centers[s1_index - 1][:2]
            used_indices.add(s1_index - 1)

        for i in range(1, 5):
            idx = s1_index + i
            if idx < len(centers):
                annotations[f"S{i+1}"] = centers[idx][:2]
                used_indices.add(idx)

        for i in range(1, 5):
            idx = s1_index - i - 1
            if idx >= 0:
                annotations[f"L{5 - i}"] = centers[idx][:2]
                used_indices.add(idx)


        # 反向補標邏輯（L1→L2→L3→L4→L5）
        label = "L1"
        while label in annotations:
            next_num = int(label[1]) + 1
            if next_num > 5:
                break
            next_label = f"L{next_num}"
            x_ref, y_ref = annotations[label]
            best_match = None
            min_dist = float('inf')
            for i, (cx, cy, box) in enumerate(centers):
                if i in used_indices or cx <= x_ref:
                    continue
                dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                if dist < min_dist and dist < shift_threshold:
                    min_dist = dist
                    best_match = (i, cx, cy, box)
            if best_match:
                i, cx, cy, box = best_match
                annotations[next_label] = (cx, cy)
                used_indices.add(i)
                label = next_label
            else:
                break

        # 反向補標邏輯（S5→S4→S3→S2→S1）
        label = "S5"
        while label in annotations:
            next_num = int(label[1]) - 1
            if next_num < 1:
                break
            next_label = f"S{next_num}"
            x_ref, y_ref = annotations[label]
            best_match = None
            min_dist = float('inf')
            for i, (cx, cy, box) in enumerate(centers):
                if i in used_indices or cx >= x_ref:
                    continue
                dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                if dist < min_dist and dist < shift_threshold:
                    min_dist = dist
                    best_match = (i, cx, cy, box)
            if best_match:
                i, cx, cy, box = best_match
                annotations[next_label] = (cx, cy)
                used_indices.add(i)
                label = next_label
            else:
                break

        # 延伸胸椎 T12~T1（從 L1 左邊開始標）
        label = "L1"
        if label in annotations:
            x_ref, y_ref = annotations[label]
            for t_num in reversed(range(1, 13)):  # T12 → T1
                next_label = f"T{t_num}"
                best_match = None
                min_dist = float('inf')
                for i, (cx, cy, box) in enumerate(centers):
                    if i in used_indices or cx >= x_ref - 5:  # 稍微放寬範圍
                        continue
                    dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                    if dist < min_dist and dist < 60:  # 放寬 shift_threshold 為 60
                        min_dist = dist
                        best_match = (i, cx, cy, box)
                if best_match:
                    i, cx, cy, box = best_match
                    annotations[next_label] = (cx, cy)
                    used_indices.add(i)
                    x_ref, y_ref = cx, cy  # 更新參考點為新標註位置
                else:
                    break  # 若無可用框則停止延伸

    # 補標遺失節點（再檢查一次）
    annotations = auto_label_missing_bones(annotations, centers, used_indices)

    # 畫框避免重疊
    drawn_positions = []
    for label, (cx, cy) in annotations.items():
        for x, y, (x1, y1, bw, bh) in centers:
            if cx == x and cy == y:
                too_close = any(abs(px - cx) < 10 and abs(py - cy) < 10 for px, py in drawn_positions)
                if too_close:
                    break
                drawn_positions.append((cx, cy))
                cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 255), 2)
                cv2.putText(img, label, (x1, y1 + bh + 30), font, 1, (0, 255, 255), 2)

    # 畫出未使用的框
    for i, (cx, cy, (x1, y1, bw, bh)) in enumerate(centers):
        if i not in used_indices:
            too_close = any(abs(px - cx) < 10 and abs(py - cy) < 10 for px, py in drawn_positions)
            if too_close:
                continue  # 跳過重疊太近的紅框
            drawn_positions.append((cx, cy))
            cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)
    return annotations

# 處理所有圖片
for image_path in image_paths:
    txt_path = image_path.replace(".jpg", ".txt")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    frame_number = basename.split("_")[1]
    output_name = f"track_{frame_number}.jpg"
    output_path = os.path.join(output_dir, output_name)

    if not os.path.exists(txt_path):
        print(f"缺少對應座標檔：{os.path.basename(txt_path)}，跳過")
        continue

    # 取得目前幀的標註資訊，參考前 10 幀
    prev_info = list(recent_annotations)[::-1]  # 最新的在前面
    annotations = annotate_bones(image_path, txt_path, output_path, prev_info)

    if annotations is None or len(annotations) == 0:
        empty_frame_count += 1
        if empty_frame_count >= 10:
            print("停止：連續十幀無偵測框，系統暫停")
            break
    else:
        empty_frame_count = 0  # 成功偵測，清空失幀計數
        recent_annotations.append(annotations)  # 加入目前幀結果