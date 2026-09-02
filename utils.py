import cv2
import numpy as np
import os
import json

# --- 移除近乎重疊的框 (根據 IOU) ---
def filter_overlapping_boxes(centers, labels_dict, iou_threshold=0.8):
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    keep = []
    removed = set()
    for i in range(len(centers)):
        if i in removed:
            continue
        cx1, cy1, box1 = centers[i]
        keep_i = True
        for j in range(i + 1, len(centers)):
            if j in removed:
                continue
            cx2, cy2, box2 = centers[j]
            iou = compute_iou(box1, box2)
            if iou > iou_threshold:
                # 優先保留已標註的
                label1 = any((cx1, cy1) == v for v in labels_dict.values())
                label2 = any((cx2, cy2) == v for v in labels_dict.values())
                if label1 and not label2:
                    removed.add(j)
                elif label2 and not label1:
                    removed.add(i)
                    keep_i = False
                    break
                else:
                    removed.add(j)
        if keep_i:
            keep.append(centers[i])
    return keep

# YOLO txt 轉中心點與邊框 (cx, cy, [x1, y1, bw, bh])
def yolo_to_centers(txt_path, img_shape):
    centers = []
    h, w = img_shape[:2]
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, bw, bh = map(float, parts)
            cx, cy = int(x * w), int(y * h)
            bw, bh = int(bw * w), int(bh * h)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            centers.append((cx, cy, (x1, y1, bw, bh)))
    centers.sort(key=lambda c: -c[0])
    return centers

# 主標註函式（第一幀初始化 / 後續追蹤）
def annotate_bones_count(image_path, txt_path, output_path,
                         prev_annotations_list=None, disappeared_labels=None,
                         max_move_threshold=50):
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        centers = yolo_to_centers(txt_path, img.shape)
    except Exception as e:
        print(f"無法讀取 {os.path.basename(txt_path)}")
        return None
    if not centers:
        return None
    labels_dict = {}
    used_indices = set()
    method_dict = {}

    disappeared_from_prev = {}
    if prev_annotations_list:
        last_frame = prev_annotations_list[-1]
        for label, (px, py) in last_frame.items():
            found = False
            for cx, cy, _ in centers:
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < max_move_threshold:
                    found = True
                    break
            if not found:
                disappeared_from_prev[label] = (px, py)

    # 自動估計 max_move_threshold
    if len(centers) >= 2:
        # 取前兩個橫向相鄰骨頭的距離估算
        dists = [np.linalg.norm(np.array(centers[i][0:2]) - np.array(centers[i+1][0:2])) 
                 for i in range(len(centers)-1)]
        avg_dist = np.mean(dists)
        max_move_threshold = max(30, min(80, avg_dist * 0.8))  # 下限30，上限80


    if prev_annotations_list:  # 改進的追蹤邏輯
        candidate_matches = []
        for prev_frame in prev_annotations_list:
            for label, (px, py) in prev_frame.items():
                if label in labels_dict:
                    continue
                for i, (cx, cy, box) in enumerate(centers):
                    if i in used_indices:
                        continue
                    dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if dist < max_move_threshold:
                        candidate_matches.append((dist, label, i, (cx, cy)))

        # 依距離排序後配對最近者
        candidate_matches.sort()
        matched_labels = set()
        matched_indices = set()
        for dist, label, i, (cx, cy) in candidate_matches:
            if label in matched_labels or i in matched_indices:
                continue

            past_positions = []
            for past_frame in prev_annotations_list:
                if label in past_frame:
                    past_positions.append(past_frame[label])
            if len(past_positions) >= 3:
                avg_px = sum(p[0] for p in past_positions) / len(past_positions)
                avg_py = sum(p[1] for p in past_positions) / len(past_positions)
                dist_to_avg = np.sqrt((cx - avg_px) ** 2 + (cy - avg_py) ** 2)
                if dist_to_avg > max_move_threshold:
                    continue

            # 限制：若 label-1 已存在，則此 label 的中心 x 要更靠左（小）
            if label - 1 in labels_dict:
                prev_cx, _ = labels_dict[label - 1]
                if cx > prev_cx:
                    continue

            # 限制：若 label+1 已存在，則此 label 的中心 x 要更靠右（大）
            if label + 1 in labels_dict:
                next_cx, _ = labels_dict[label + 1]
                if cx < next_cx:
                    continue

            labels_dict[label] = (cx, cy)
            method_dict[label] = "track"
            used_indices.add(i)
            matched_labels.add(label)
            matched_indices.add(i)

    if not labels_dict:  # 第一幀初始化
        for idx, (cx, cy, box) in enumerate(sorted(centers, key=lambda c: -c[0])):
            labels_dict[idx + 1] = (cx, cy)
            method_dict[idx + 1] = "init"
            used_indices.add(idx)

    # 移除重疊框
    centers[:] = filter_overlapping_boxes(centers, labels_dict)
   # ✅ Aggressive: 強制依右到左順序重新編號，紅框變黃框、跳號效果啟用
    force_sequential_from_right(labels_dict, centers, used_indices)
    return labels_dict, centers, used_indices


# 補標規則：兩側延伸 + 中間補空
def auto_fill_missing_nodes_count(labels_dict, centers, used_indices,
                                  max_move_threshold=50,
                                  prev_annotations_list=None,
                                  current_frame_index=None,
                                  method_dict=None):
    updated = True

    if method_dict is None:
        method_dict = {}

    def estimate_avg_distance():
        sorted_labels = sorted(labels_dict.keys())
        if len(sorted_labels) < 2:
            return 50
        dists = []
        for i in range(len(sorted_labels) - 1):
            x1, y1 = labels_dict[sorted_labels[i]]
            x2, y2 = labels_dict[sorted_labels[i + 1]]
            dists.append(np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
        return np.mean(dists)

    def dynamic_distance_threshold(label_a, label_b):
        x1, y1 = labels_dict[label_a]
        x2, y2 = labels_dict[label_b]
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        avg = dist / abs(label_a - label_b)
        return avg * 0.5, avg * 1.8

    avg_dist = estimate_avg_distance()
    max_allow_dist = avg_dist * 1.6
    max_move_threshold = max(30, min(80, avg_dist * 0.8))  # 與標註階段統一邏輯

    # --- 單側延伸 (左與右) ---
    extended = True
    while extended:
        extended = False
        label_positions = {label: pos for label, pos in labels_dict.items()}
        sorted_labels = sorted(label_positions.keys())

        # 向左（遞增數字）
        for label in sorted_labels:
            x_ref, y_ref = label_positions[label]
            next_label = label + 1
            if next_label in labels_dict:
                continue
            best_match = None
            min_dist = float('inf')
            for k, (cx, cy, box) in enumerate(centers):
                if k in used_indices or cx >= x_ref:
                    continue
                dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                if dist < min_dist and dist <= max_allow_dist:
                    best_match = k
                    min_dist = dist
            if best_match is not None:
                cx, cy, _ = centers[best_match]
                labels_dict[next_label] = (cx, cy)
                used_indices.add(best_match)
                method_dict[next_label] = "left_extend"
                extended = True
                break  # 重新排序再跑下一輪
        
        # 成功補上左或右節點後，重新排序
        sorted_labels = sorted(labels_dict.keys())

        # 向右（遞減數字）
        label_positions = {label: pos for label, pos in labels_dict.items()}
        sorted_labels = sorted(label_positions.keys())
        for label in sorted_labels[::-1]:
            x_ref, y_ref = label_positions[label]
            next_label = label - 1
            if next_label in labels_dict or next_label <= 0:
                continue
            best_match = None
            min_dist = float('inf')
            for k, (cx, cy, box) in enumerate(centers):
                if k in used_indices or cx <= x_ref:
                    continue
                dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                if dist < min_dist and dist <= max_allow_dist:
                    best_match = k
                    min_dist = dist
            if best_match is not None:
                cx, cy, _ = centers[best_match]
                labels_dict[next_label] = (cx, cy)
                used_indices.add(best_match)
                method_dict[next_label] = "right_extend"
                extended = True
                break

        # 成功補上左或右節點後，重新排序
        sorted_labels = sorted(labels_dict.keys())

    # --- 中間插補，允許差值 > 2 ---
    inserted = True
    while inserted:
        inserted = False
        label_positions = {label: pos for label, pos in labels_dict.items()}
        sorted_labels = sorted(label_positions.keys())
        for i in range(len(sorted_labels) - 1):
            l1, l2 = sorted_labels[i], sorted_labels[i + 1]
            gap = abs(l2 - l1)
            if gap <= 1:
                continue
            x1, y1 = label_positions[l1]
            x2, y2 = label_positions[l2]
            min_dist, max_dist = dynamic_distance_threshold(l1, l2)

            for step in range(1, gap):
                mid_label = l1 + step
                if mid_label in labels_dict:
                    continue
                mid_x = int(x1 + (x2 - x1) * (step / gap))
                mid_y = int(y1 + (y2 - y1) * (step / gap))
                best_match = None
                min_found_dist = float("inf")
                for k, (cx, cy, box) in enumerate(centers):
                    if k in used_indices:
                        continue
                    dist = np.sqrt((cx - mid_x) ** 2 + (cy - mid_y) ** 2)
                    if min_dist <= dist <= max_dist and dist < min_found_dist:
                        best_match = k
                        min_found_dist = dist
                if best_match is not None:
                    cx, cy, _ = centers[best_match]
                    labels_dict[mid_label] = (cx, cy)
                    used_indices.add(best_match)
                    method_dict[mid_label] = "recursive_fill"
                    inserted = True
                    break  # 重新排序再跑下一輪

    # --- 插補消失點（前後幀皆有） ---
    if prev_annotations_list and current_frame_index is not None and 1 <= current_frame_index < len(prev_annotations_list) - 1:
        prev_labels = prev_annotations_list[current_frame_index - 1]
        next_labels = prev_annotations_list[current_frame_index + 1]
        common_keys = set(prev_labels.keys()) & set(next_labels.keys())
        for label in common_keys:
            if label in labels_dict:
                continue
            x1, y1 = prev_labels[label]
            x2, y2 = next_labels[label]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            best_match = None
            min_dist = float('inf')
            for k, (cx, cy, box) in enumerate(centers):
                if k in used_indices:
                    continue
                dist = np.sqrt((cx - mid_x)**2 + (cy - mid_y)**2)
                if dist <= max_allow_dist and dist < min_dist:
                    best_match = k
                    min_dist = dist
            if best_match is not None:
                cx, cy, _ = centers[best_match]
                labels_dict[label] = (cx, cy)
                used_indices.add(best_match)
                method_dict[label] = "recover"

def force_sequential_from_right(labels_dict, centers, used_indices):
    """
    依『由右到左』的 x 座標順序，強制把所有偵測框連號指派：
    第 1 個（最右）→ label=1、接著 2、3、…
    - 原本已標註的節點會被「重編號」（例如原先=5，若中間插入2個紅框，則會變=7）
    - 原本紅框（未使用）也會拿到新編號，並被加入 used_indices
    這樣就能達成你要的『看到紅框就跳號（往左位移）』的效果。
    """
    # 由右到左排序（centers 的結構：[(cx, cy, (x1, y1, w, h)), ...]）
    ordered = sorted([(i, cx, cy) for i, (cx, cy, _) in enumerate(centers)], key=lambda t: -t[1])

    new_labels = {}
    # 逐一（右→左）連號
    for k, (idx, cx, cy) in enumerate(ordered, start=1):
        new_labels[k] = (cx, cy)
        used_indices.add(idx)  # 這行會把原本紅框也視為「已使用」

    labels_dict.clear()
    labels_dict.update(new_labels)


# 繪圖並輸出結果，數字顯示在框下方
def draw_annotations(image_path, labels_dict, centers, used_indices, output_path):
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    drawn_positions = []

    # 畫已標註框（黃色）與數字（下方）
    for label, (cx, cy) in labels_dict.items():
        for (bx, by, (x1, y1, bw, bh)) in centers:
            if (cx, cy) == (bx, by):
                too_close = any(np.sqrt((cx - px)**2 + (cy - py)**2) < 10 for (px, py) in drawn_positions)
                if too_close:
                    continue
                drawn_positions.append((cx, cy))
                cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 255), 2)
                cv2.putText(img, str(label), (x1, y1 + bh + 20), font, 0.6, (0, 255, 255), 2)

    # 畫未使用框（紅色）
    for i, (cx, cy, (x1, y1, bw, bh)) in enumerate(centers):
        if i in used_indices:
            continue
        too_close = any(np.sqrt((cx - px)**2 + (cy - py)**2) < 10 for (px, py) in drawn_positions)
        if too_close:
            continue
        drawn_positions.append((cx, cy))
        cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)

def write_nodes_txt(nodes_txt_path, labels_dict):
    """
    寫出節點清單：
      count: <N>
      nodes: <label1> <label2> ...
      <label> <cx> <cy>
      ...
    labels_dict: {label(int/str): (cx, cy)}
    """
    try:
        def _to_int(lbl):
            try:
                return int(lbl)
            except Exception:
                return 10**6
        items = sorted(labels_dict.items(), key=lambda kv: _to_int(kv[0]))
        labels = [str(k) for k, _ in items]
        with open(nodes_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"count: {len(labels)}\n")
            f.write("nodes: " + " ".join(labels) + "\n")
            for k, (cx, cy) in items:
                f.write(f"{k} {int(cx)} {int(cy)}\n")
        return True
    except Exception:
        try:
            with open(nodes_txt_path, 'w', encoding='utf-8') as f:
                f.write("count: 0\nnodes: \n")
        except Exception:
            pass
        return False

def normalize_spine_labels(labels_dict, class_to_name=None):
    """
    將 labels_dict 的 key 正規化成解剖標籤（L1~L5, S1~S5）。
    - labels_dict: {label: (cx, cy)}
    - class_to_name: 例如 {0:"L1",1:"L2",2:"L3",3:"L4",4:"L5",5:"S1"...}
      如果 None，則假設 labels_dict 本來就已經是 "L5" 這種字串。
    回傳: new_dict (key 一定是字串)
    """
    new_dict = {}
    for k, v in labels_dict.items():
        if class_to_name is not None:
            # 你的 track 模式若用 class id，會走這裡
            try:
                kk = class_to_name[int(k)]
            except Exception:
                kk = str(k)
        else:
            kk = str(k)

        new_dict[str(kk)] = (int(v[0]), int(v[1]))
    return new_dict

def write_nodes_json(json_path, labels_dict):
    """
    將標註結果輸出為 JSON，格式類似 Lumbar_2_contour.json：
    {
        "L2": {"x": 161, "y": 227},
        "L3": {"x": 266, "y": 223},
        ...
    }
    labels_dict: {label(int/str): (cx, cy)}
    """
    try:
        data = {}
        for label, (cx, cy) in labels_dict.items():
            # label 轉成字串，座標轉 int
            data[str(label)] = {
                "x": int(cx),
                "y": int(cy)
            }

        dir_name = os.path.dirname(json_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return True
    except Exception:
        return False

def write_log_entry(log_path, image_name, original_centers, sorted_centers, labels_dict, centers, used_indices, method_dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"📄 {image_name}.jpg\n\n")

        f.write("[原始 YOLO 框中心點座標]\n")
        for cx, cy, _ in original_centers:
            f.write(f"({cx}, {cy})\n")

        f.write("\n[右到左排序後中心點座標]\n")
        for cx, cy, _ in sorted_centers:
            f.write(f"({cx}, {cy})\n")

        f.write("\n[標註結果：label -> 中心點座標 + 框 + 方法]\n")
        for label in sorted(labels_dict.keys(), key=int):
            cx, cy = labels_dict[label]
            matched_box = None
            for i, (px, py, box) in enumerate(centers):
                if (px, py) == (cx, cy):
                    matched_box = box
                    break
            if matched_box:
                x1, y1, w, h = matched_box
                method = method_dict.get(label, "unknown")
                f.write(f"{label}: center=({cx},{cy}), box=({x1},{y1},{w},{h}), method={method}\n")

        unused = [i for i in range(len(centers)) if i not in used_indices]
        f.write(f"\n[未使用的框 index 共 {len(unused)} 個]\n")
        if unused:
            for i in unused:
                cx, cy, _ = centers[i]
                f.write(f"index {i}: ({cx}, {cy})\n")
        else:
            f.write("無\n")

        if len(labels_dict) > 1:
            label_nums = sorted([int(k) for k in labels_dict])
            diffs = [label_nums[i+1] - label_nums[i] for i in range(len(label_nums)-1)]
            discontinuity = [i for i, d in enumerate(diffs) if d != 1]
            if discontinuity:
                f.write(f"\n[跳號檢查] ⛔ 發現跳號：{[(label_nums[i], label_nums[i+1]) for i in discontinuity]}\n")
            else:
                f.write("\n[跳號檢查] ✅ 標註連續\n")
        else:
            f.write("\n[跳號檢查] 標註不足，無法檢查\n")

        f.write("\n" + "-" * 40 + "\n\n")