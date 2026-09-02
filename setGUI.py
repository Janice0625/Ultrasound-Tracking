# 標準函式庫 (Python stdlib)
import os                 # 檔案/路徑
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
import re                 # 正則表達式
import time               # 時間工具 (sleep、timestamp)
import csv                # 讀寫 CSV
import json               # 讀寫 JSON
import ctypes             # 呼叫 DLL / C 介面
import sys
import io
import logging
import datetime           # 日期時間格式化
import shutil as sh       # 檔案/資料夾複製、搬移
import threading          # 執行緒/同步基礎
from threading import Event, Thread, Lock  # 事件旗標、執行緒、互斥鎖
from collections import deque              # 雙端佇列 (追蹤/前幀緩存用)
import glob              # 檔案通配搜尋 (glob.glob)
from glob import glob    # 直接使用函式 glob(...)（※如要統一風格，可只留一種）

# GUI (PyQt6)
from PyQt6.QtGui import QPixmap
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QTimer, QTime
from PyQt6.QtWidgets import QApplication

# 影像處理
from PIL import Image, ImageGrab   # 影像 I/O、螢幕截圖
import cv2                         # OpenCV：電腦視覺處理
import numpy as np                 # NumPy：陣列/向量運算

# 深度學習 / 推論
import torch                       # PyTorch：GPU/張量運算（例如 YOLO 的裝置偵測）
import tensorflow as tf            # TensorFlow：Keras 模型推論
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import segmentation_models as sm   # segmentation-models：提供 U-Net/損失/指標
from ultralytics import YOLO       # Ultralytics YOLO：偵測/標註匯出

# 專案內部工具引用
from utils import (
    annotate_bones_count,          # 計數模式：標註骨節點
    auto_fill_missing_nodes_count, # 自動補標遺失節點
    draw_annotations,              # 把標註結果畫回影像
    write_log_entry,               # 紀錄/寫入 log
    write_nodes_txt,               # nodes_txt
    write_nodes_json               # ★ 新增：JSON 版座標檔
)

# Define image width and height
image_height = 600
image_width = 480
SIZE = (image_height, image_width)
#判斷當前系統是否支援 GPU 計算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys, traceback

def excepthook(exc_type, exc_value, exc_traceback):
    """
    統一處理未捕捉例外：
    只寫到「原生 stderr」，不要再經過 Qt 訊號或被我們後面重導的 sys.stderr 影響。
    這樣就算 Qt 或 StreamToEmitter 出問題，至少錯誤會確實印在 console。
    """
    try:
        sys.__stderr__.write("\n========== 💥 Unhandled Exception 💥 ==========\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.__stderr__)
        sys.__stderr__.write("==============================================\n")
        sys.__stderr__.flush()
    except Exception:
        # 如果連 __stderr__ 都無法寫，就放棄，避免再噴更多錯
        pass

sys.excepthook = excepthook

# 處理圖像
def preprocess_image(image, size):
    resized_image = cv2.resize(image, (size[1], size[0]))
    # 將圖像的像素值從整數 (0-255) 轉換為浮點數 (0.0-1.0)，以便標準化輸入數據
    resized_image = resized_image.astype('float32') / 255.0
    #OpenCV 的圖像(高度, 寬度, 通道)-> PyTorch 模型(通道, 高度, 寬度)
    images = np.transpose(resized_image, (2, 0, 1))
    images = np.expand_dims(images, axis=0)  # 添加批次維度
    images = torch.from_numpy(images)
    images = images.to(device)
    return images

def dice_loss_plus_focal_loss(y_true, y_pred):
    """
    U-Net 自訂義 Loss (Dice Loss + Focal Loss)
    """
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

class LogEmitter(QtCore.QObject):
    message = QtCore.pyqtSignal(str)
    show_image = QtCore.pyqtSignal(str, str)  
    # 例如: show_image.emit("frame6", output_path) / ("rt_lumbar_preview", path)

class StreamToEmitter(io.TextIOBase):
    """把 sys.stdout / sys.stderr 重導到 Qt 訊號。"""
    def __init__(self, emit_signal: QtCore.pyqtSignal, prefix: str = ""):
        super().__init__()
        self.emit_signal = emit_signal
        self._buffer = ""
        self.prefix = prefix  # 給 stderr 用 (例如 "[ERR] ")

    def _safe_emit(self, text: str):
        """
        安全發送訊息到 Qt 訊號。
        如果訊號處理過程又丟例外，不要再讓程式整個炸掉。
        """
        try:
            self.emit_signal.emit(text)
        except Exception:
            # 若 UI 已經被關掉，或訊號處理出錯，就退回原本的 stderr（如果還在）
            try:
                sys.__stderr__.write(text + "\n")
                sys.__stderr__.flush()
            except Exception:
                # 連原生 stderr 都不能用，就直接吞掉，避免無限遞迴
                pass

    def write(self, s: str):
        # 將資料切成行，逐行送出（避免半行）
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._safe_emit(self.prefix + line)
        return len(s)

    def flush(self):
        if self._buffer.strip():
            self._safe_emit(self.prefix + self._buffer.rstrip("\r\n"))
        self._buffer = ""

class GUI:
    def __init__(self, app, root):
        # 0) 環境 / 裝置設定（TensorFlow GPU 記憶體策略）
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"設定 GPU 記憶體增長失敗: {e}")

        # 基本參考 / 視窗物件
        # 以螢幕截圖作為影像來源的座標與尺寸（左上 x,y 與寬高）
        self.x, self.y, self.width, self.height = 300, 230, 410, 410
        self.app, self.root = app, root

        # 是否在訊息前面加時間戳（你說不要時間戳，所以預設 False）
        self.LOG_WITH_TIME = False

        # 錄影狀態 / 執行緒同步
        self.save = True                         # 影像擷取主迴圈開關
        self.save_dir = "saved_images"
        self.recording = False
        self.stop_recording_event = Event()   # ← 錄影用
        self.stop_track_event = Event()       # ← 追蹤用
        self.is_tracking = False
        self.stop_count_event = Event()
        self.is_counting = False

        self.video_writer = None                 # 影片寫入器（如需輸出影片）
        self.record_thread = None                # 錄影執行緒
        self.record_timer = QTimer(self.root)    # 記錄「錄影時間」的 UI 計時器
        self.record_timer.timeout.connect(self.update_time)
        self.time = QTime(0, 0, 0)               # 錄影時間初始值 00:00:00

        # 其他同步/旗標
        self.stop_update_event = Event()         # 可用於停止某些更新流程
        self.lock = threading.Lock()             # 共享資源的互斥鎖
        self.recorded_rotations = None           # 若有紀錄旋轉矩陣的需求
        self.stop_update_calibration_label = False  # 停止顯示校正數字
        self.fixed_arrows = None                 # 例如固定的箭頭繪製資訊
        self.sensor_result = " "                 # 感測器檢測結果文字
        self.slope = 1000000                     # 初始斜率（設極大值）

        # 位置預測 / 偵測工作狀態
        self.locating = False
        self.stop_location_event = Event()
        self.predicted_label = " "
        self.location_thread = None

        # 影像緩衝（例如螢幕截圖循環緩存）
        self.buffer_size = 3
        self.screen_buffer = []

        # UI 狀態（頁籤 / 子模式）
        # mode：'stitch'（拼接）或 'realtime'（即時）
        self.mode = "stitch"
        # 即時子模式：'track'（追蹤）或 'count'（計數）
        self.rt_submode = "track"

        # 6) 樣式（StyleSheet：按鈕 / 標題 / 框線）
        self.button_style = """
            QPushButton {
                border: 2px solid black;
                border-radius: 20px;
                padding: 5px 12px;
                color: black;
                background-color: white;
            }
            QPushButton:disabled {
                border-color: gray;
                color: gray;
                background-color: #F0F0F0;
            }
            QPushButton:hover { border: 2px solid #555; background-color: #F5F5F5; }
            QPushButton:pressed { background-color: #E0E0E0; }
        """
        self.active_button_style = """
            QPushButton {
                border: 2px solid black;
                border-radius: 20px;
                padding: 5px 12px;
                color: black;
                background-color: #EAEAEA;   /* 被選中的灰底 */
            }
        """
        self.caption_style = """
            QLabel {
                background-color: white;
                border: 2px solid black;
                padding: 2px 8px;
                color: black;
                font-size: 14px;
                font-family: 'Microsoft JhengHei';
            }
        """
        self.box_style = """
            background-color: white;
            border: 2px solid black;
            font-size: 18px;
            font-family: 'Microsoft JhengHei';
            color: #333333;
            qproperty-alignment: AlignCenter;
        """

        # 進階樣式（統一風格）
        self.BUTTON_STYLE = """
        QPushButton {
            border: 2px solid #111;
            border-radius: 18px;
            padding: 6px 14px;
            background: #FFFFFF;
            color: #111;
        }
        QPushButton:hover  { background: #F6F6F6; }
        QPushButton:pressed{ background: #EDEDED; }
        QPushButton:disabled { color:#888; border-color:#888; background:#F5F5F5; }
        QPushButton#active { background:#EAEAEA; border-width:3px; }  /* 高亮選中 */
        """
        self.FRAME_STYLE = """
            background-color: #FFFFFF;
            border: 2px solid #111;
            border-radius: 0px; /* 影像框直角 */
        """
        self.CAPTION_STYLE = """
        QLabel {
            background:#FFF; color:#111; border:2px solid #111;
            padding:2px 8px; font-size:14px;
        }
        """
        self.LABEL_STYLE = "color:#111; background:transparent;"

        # === 初始化 Qt log emitter ===
        # 先把「當前的」stdout / stderr 存起來，等等真的要寫 console 可以用
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._log_emitter = LogEmitter(self.root)            # 建立訊號物件（只建立一次）
        self._log_emitter.message.connect(self._append_log)  # 連到 UI append 函式
        self._log_emitter.show_image.connect(self._on_show_image)

        # 先不要在這裡把全域的 stdout / stderr 重導到 Qt，
        # 這是最容易在啟動階段 + 例外時造成「訊號裡又丟例外 → 無限遞迴 → 直接炸掉」的地方
        #
        # 如果之後確認 GUI 正常、想再把 print 也顯示在右下訊息框，
        # 再慢慢打開下面兩行做測試。
        #
        # sys.stdout = StreamToEmitter(self._log_emitter.message)
        # sys.stderr = StreamToEmitter(self._log_emitter.message, prefix="[ERR] ")

        # 版面常數（Margin / Gutter / Y 位置）
        self.M = 20      # 外距 margin
        self.G = 20      # 元件水平/垂直間距 gutter
        self.T1 = 30     # 第一排按鈕與標籤的 y
        self.T2 = 80     # 第一排第二列 y（例如時間）
        self.Y_FRAMES = 210  # 影像框群組起始 y

        # 建立 UI 元件（框、按鈕、標籤）
        #    ※ 順序重要：先 interface() 佈局 → button() 建按鈕 → label() 建文字
        self.interface()
        self.button()
        self.label()

        # 啟動影像擷取（背景執行）
        self.start()
        time.sleep(0.5)                # 給擷取執行緒一點啟動時間

        # 頁面顯示初始化（依目前 self.mode 切換顯示/隱藏）
        self.set_mode(self.mode)

        # 模型 / 資料根目錄（保留下來供路徑縮短用）
        BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "models")
        self.BASE_DIR  = BASE_DIR
        self.MODEL_DIR = MODEL_DIR

        self.DATA_ROOT = os.path.join(BASE_DIR, "data")
        os.makedirs(self.DATA_ROOT, exist_ok=True)

        # 是否在訊息前面加時間戳；你要求不要時間戳，所以設 False
        self.LOG_WITH_TIME = False

        # Keras U-Net：使用自訂義物件載入（避免 segmentation 階段重複載入）
        self.lumbar_model = self.load_model_with_custom_objects(
            os.path.join(MODEL_DIR, "lumbar_unet_model.h5")
        )
        self.sacrum_model = self.load_model_with_custom_objects(
            os.path.join(MODEL_DIR, "sacrum_unet_model.h5")
        )
        # Ultralytics YOLO 權重
        self.yolo_model = YOLO(os.path.join(MODEL_DIR, "best.pt")).to('cuda')
        self.log(f"YOLO 強制使用 GPU: {self.yolo_model.device}")

        self.rt_use_ram = True          # 走 RAM 加速管線
        self.mirror_to_disk = True      # 同步鏡射到資料夾，維持相容
        self.frame_queue = deque(maxlen=128)   # 影像佇列 (fid, np.ndarray)
        self.label_store = {}                  # fid -> "yolo txt 內容 (str)"
        self.next_frame_id = 0
        self._yolo_stop = Event()
        self._yolo_thread = None

        self._track_thread = None            # 保存追蹤執行緒

        self._init_paths()
        
    def _on_show_image(self, target: str, path: str):
        lab = getattr(self, target, None)
        if isinstance(lab, QtWidgets.QLabel) and os.path.exists(path):
            pix = QtGui.QPixmap(path).scaled(
                lab.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            lab.setPixmap(pix)
            lab.setScaledContents(False)

    def log(self, text: str, level: str = "info"):
        """
        統一 log 入口：
        - text：要顯示的訊息
        - level：'info' / 'error' 等，目前只是保留欄位，不影響顯示
        """
        msg = str(text)

        # 先寫到 UI（若 _log_emitter 已經建立）
        if hasattr(self, "_log_emitter"):
            try:
                self._log_emitter.message.emit(msg)
            except Exception:
                # UI 若還沒準備好，就先忽略，不要讓程式當掉
                pass

        # 也保留到原本終端機（方便在 IDE / console 看到）
        if hasattr(self, "_orig_stdout"):
            try:
                self._orig_stdout.write(msg + "\n")
                self._orig_stdout.flush()
            except Exception:
                pass

    def _shorten_paths(self, s: str) -> str:
        """把絕對路徑縮短成以 data/ 或 models/ 開頭的相對顯示。"""
        try:
            # 先標準化分隔符，避免混用斜線
            s_norm = s.replace("\\", os.sep).replace("/", os.sep)

            # 依優先順序替換（先 data 再 models；BASE_DIR 最後）
            if self.DATA_ROOT:
                s_norm = s_norm.replace(self.DATA_ROOT + os.sep, f"data{os.sep}")
            if self.MODEL_DIR:
                s_norm = s_norm.replace(self.MODEL_DIR + os.sep, f"models{os.sep}")
            if self.BASE_DIR:
                # 若還有殘餘 BASE_DIR 前綴，直接砍到專案根（如想保留可改成 ".")
                s_norm = s_norm.replace(self.BASE_DIR + os.sep, "")

            # 還原到系統的分隔符
            return s_norm
        except Exception:
            return s

    def _append_log(self, text: str):
        """
        把文字加到訊息框：不帶時間戳；自動縮短路徑；逐行輸出；自動捲到底。
        這個函式會被 self.log() 與 stdout/stderr 重導呼叫。
        """
        if text is None:
            return

        # 轉字串與清理換行
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        for line in text.split("\n"):
            if not line:
                continue

            # 過去若有殘留的 [HH:MM:SS] 前綴，順手移除
            line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)

            # 路徑縮短
            line = self._shorten_paths(line)

            # 是否加時間戳（你現在不要；LOG_WITH_TIME 若還沒建好也不要炸）
            if getattr(self, "LOG_WITH_TIME", False):
                ts = time.strftime("%H:%M:%S")
                line = f"[{ts}] {line}"

            # 如果 msg_console 還沒建立（介面還在初始化），就先寫回終端機，不要當掉
            if not hasattr(self, "msg_console"):
                try:
                    if hasattr(self, "_orig_stdout"):
                        self._orig_stdout.write(line + "\n")
                        self._orig_stdout.flush()
                except Exception:
                    pass
                continue

            # 正常情況：寫到 UI 的訊息框
            self.msg_console.appendPlainText(line)

        # 捲到最底（前提是 msg_console 已經存在）
        if hasattr(self, "msg_console"):
            sb = self.msg_console.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _init_paths(self):
        """
        初始化路徑設定。務必在任何呼叫 self._p(...) 之前執行。
        """
        import os
        # 專案根目錄（以這支檔案所在資料夾為準）
        self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        # data 根目錄
        self.DATA_ROOT = os.path.join(self.PROJECT_ROOT, "data")
        os.makedirs(self.DATA_ROOT, exist_ok=True)

    def _p(self, *parts):
        """
        在 DATA_ROOT 下組路徑並確保資料夾存在。
        """
        if not hasattr(self, "DATA_ROOT"):
            raise RuntimeError("DATA_ROOT 尚未初始化。請先呼叫 self._init_paths()")
        import os
        p = os.path.join(self.DATA_ROOT, *parts)
        # 最後一層若像是檔案就只建父資料夾；否則直接建自己（保守起見全部建）
        os.makedirs(p, exist_ok=True)
        return p

    def _next_index(self, dir_path: str, prefix: str, suffix: str) -> int:
        """
        在 dir_path 中找 prefix_*_suffix 的檔名，回傳下一個流水號整數。
        例如：lumbar_3_selected.jpg -> 回傳 4
        """
        try:
            names = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(suffix)]
        except FileNotFoundError:
            names = []
        max_i = 0
        for f in names:
            try:
                i = int(f.split("_")[1])
                if i > max_i:
                    max_i = i
            except Exception:
                pass
        return max_i + 1

    def _latest_by_index(self, dir_path: str, prefix: str, middle_token: str, suffix: str):
        """
        在 dir_path 中找出 prefix_{idx}_{middle_token}{suffix} 之中 idx 最大的檔案。
        回傳 (完整路徑, idx)；若無則 (None, None)
        例：prefix='lumbar_', middle_token='_preprocessed', suffix='.jpg'
        """
        if not os.path.isdir(dir_path):
            return None, None
        cand = [f for f in os.listdir(dir_path)
                if f.startswith(prefix) and f.endswith(suffix) and middle_token in f]
        if not cand:
            return None, None

        def _num(name: str) -> int:
            try:
                return int(name.split("_")[1])
            except Exception:
                return -1

        latest = max(cand, key=_num)
        idx = _num(latest)
        return (os.path.join(dir_path, latest), idx) if idx >= 0 else (None, None)

    def _latest_preprocessed(self, kind: str):
        """最新的 <lumbar|sacrum>_<idx>_preprocessed.jpg"""
        slug = kind.lower().strip()
        pre_dir = self._p("stitching", "preprocessed")
        return self._latest_by_index(pre_dir, f"{slug}_", "_preprocessed", ".jpg")

    def _latest_unet(self, kind: str):
        """最新的 <lumbar|sacrum>_<idx>_unet.png"""
        slug = kind.lower().strip()
        unet_dir = self._p("stitching", "unet")
        return self._latest_by_index(unet_dir, f"{slug}_", "_unet", ".png")

    def _latest_meta(self, kind: str):
        """最新的 <lumbar|sacrum>_<idx>_contour.json"""
        slug = kind.lower().strip()
        meta_dir = self._p("stitching", "meta")
        return self._latest_by_index(meta_dir, f"{slug}_", "_contour", ".json")

    def _latest_recording_session(self):
        """
        回傳：
        latest_dir:  <.../data/recording/YYYYMMDD/N>
        session_folder: 'YYYYMMDD_N'（拿來當作輸出 session 名稱）
        若找不到則回傳 (None, None)
        """
        rec_root = self._p("recording")  # <BASE>/data/recording
        if not os.path.isdir(rec_root):
            return None, None

        days = [d for d in os.listdir(rec_root)
                if os.path.isdir(os.path.join(rec_root, d)) and d.isdigit()]
        if not days:
            return None, None
        latest_day = max(days)

        day_dir = os.path.join(rec_root, latest_day)
        nums = [n for n in os.listdir(day_dir)
                if os.path.isdir(os.path.join(day_dir, n)) and n.isdigit()]
        if not nums:
            return None, None
        latest_n = str(max(map(int, nums)))

        latest_dir = os.path.join(day_dir, latest_n)
        session_folder = f"{latest_day}_{latest_n}"
        return latest_dir, session_folder

    def set_mode(self, mode: str):
        """切換拼接／即時模式，控制可見元件與按鈕底色"""
        if mode == self.mode:
            return
        self.mode = mode

        # 上排兩顆模式按鈕底色
        if self.mode == "stitch":
            self.btn_mode_stitch.setStyleSheet(self.active_button_style)
            self.btn_mode_rt.setStyleSheet(self.button_style)
        else:
            self.btn_mode_stitch.setStyleSheet(self.button_style)
            self.btn_mode_rt.setStyleSheet(self.active_button_style)

        # ——— 顯示／隱藏：拼接模式的功能鍵 ———
        stitch_widgets = [
            self.lumbar_input_btn, self.lumbar_pretreatment_btn,
            self.sacral_input_btn, self.sacral_pretreatment_btn,
            self.segmentation_btn, self.contour_btn, self.splicing_btn
        ]
        for w in stitch_widgets:
            w.setVisible(self.mode == "stitch")

        # ——— 顯示／隱藏：拼接模式的影像框與小標題 ———
        for w in [self.frame1, self.frame2, self.frame3, self.frame4, self.frame5,
                  self.cap_lumbar, self.cap_sacrum, self.cap_stitch2d,
                  self.cap_lumbar_unet, self.cap_sacrum_unet]:
            w.setVisible(self.mode == "stitch")

        # ——— 顯示／隱藏：即時模式的影像框與小標題、子模式按鈕 ———
        for w in [
            self.frame6, self.frame7, self.cap_rt, self.cap_rt_stitch,
            self.btn_rt_track, self.btn_rt_count, self.sep_label,
            self.rt_lumbar_preview, self.rt_sacrum_preview,
            self.cap_rt_lumbar_pick, self.cap_rt_sacrum_pick,
            # 追蹤子模式的三顆 + 計數子模式的一顆，都交給 set_rt_submode 再細分顯示
            self.btn_rt_track_start, self.btn_rt_track_best1,self.btn_rt_track_best2, self.btn_rt_track_stitch,
            self.btn_rt_count_input,self.btn_rt_count_stop
        ]:
            w.setVisible(self.mode == "realtime")

        # 進入即時模式預設「追蹤」
        if self.mode == "realtime":
            self.set_rt_submode("track")

    def set_rt_submode(self, submode: str):
        """即時模式下，追蹤／計數（灰底高亮 + 顯示對應按鈕 + 控制拼接顯示框）"""
        if self.mode != "realtime":
            return
        self.rt_submode = submode

        # 更新底色
        if self.rt_submode == "track":
            self.btn_rt_track.setStyleSheet(self.active_button_style)
            self.btn_rt_count.setStyleSheet(self.button_style)
        else:
            self.btn_rt_track.setStyleSheet(self.button_style)
            self.btn_rt_count.setStyleSheet(self.active_button_style)

        # 依子模式顯示/隱藏「執行按鈕」
        is_track = (self.rt_submode == "track")
        for w in [self.btn_rt_track_start, self.btn_rt_track_stop, self.btn_rt_track_best1, self.btn_rt_track_best2, self.btn_rt_track_stitch]:
            w.setVisible(is_track)
        self.btn_rt_count_input.setVisible(not is_track)
        self.btn_rt_count_stop.setVisible(not is_track)

        # 控制即時拼接顯示框
        if is_track:
            self.frame7.show()
            if hasattr(self, "cap_rt_stitch"):
                self.cap_rt_stitch.show()
        else:
            self.frame7.hide()
            if hasattr(self, "cap_rt_stitch"):
                self.cap_rt_stitch.hide()


    def interface(self):
        # === 影像框 ===
        self.ultra_frame = QtWidgets.QLabel(self.root)
        self.ultra_frame.setGeometry(20, 230, 410, 410)
        self.ultra_frame.setStyleSheet(self.box_style)
        self.ultra_frame.setText("")  # 內容會顯示影像
        self.ultra_frame.show()

        # 拼接模式才會看到的 2D 相關框
        # 影像拼接 - 腰椎
        self.frame1 = QtWidgets.QLabel(self.root)
        self.frame1.setGeometry(480, 230, 260, 260)
        self.frame1.setStyleSheet(self.box_style)
        self.frame1.setText("")
        self.frame1.show()

        # 影像拼接 - 薦椎
        self.frame2 = QtWidgets.QLabel(self.root)
        self.frame2.setGeometry(780, 230, 260, 260)
        self.frame2.setStyleSheet(self.box_style)
        self.frame2.setText("")
        self.frame2.show()

        # 影像拼接
        self.frame3 = QtWidgets.QLabel(self.root)
        self.frame3.setGeometry(1080, 230, 450, 260)
        self.frame3.setStyleSheet(self.box_style)
        self.frame3.setText("")
        self.frame3.show()

        # 影像拼接 - 腰椎 Unet
        self.frame4 = QtWidgets.QLabel(self.root)  # 腰椎 Unet
        self.frame4.setGeometry(480, 530, 260, 260)
        self.frame4.setStyleSheet(self.box_style)
        self.frame4.setText("")
        self.frame4.show()

        # 影像拼接 - 薦椎 Unet
        self.frame5 = QtWidgets.QLabel(self.root)  # 薦椎 Unet
        self.frame5.setGeometry(780, 530, 260, 260)
        self.frame5.setStyleSheet(self.box_style)
        self.frame5.setText("")
        self.frame5.show()

        # 即時模式用的兩個大框
        # 即時處理顯示
        self.frame6 = QtWidgets.QLabel(self.root)
        self.frame6.setGeometry(480, 230, 410, 410)
        self.frame6.setStyleSheet(self.box_style)
        self.frame6.setText("")
        self.frame6.hide()  # 初始在拼接模式隱藏

        # 即時拼接顯示（完整圖）
        self.frame7 = QtWidgets.QLabel(self.root)
        self.frame7.setGeometry(1060, 230, 650, 410)
        self.frame7.setStyleSheet(self.box_style)
        self.frame7.setText("")
        self.frame7.hide()

        # ===== 即時拼接的兩個「選取預覽框」：放在 frame6 下面 =====
        # 腰椎預覽（左）
        self.rt_lumbar_preview = QtWidgets.QLabel(self.root)
        self.rt_lumbar_preview.setGeometry(480, 670, 200, 200)  # 在 frame6 下方
        self.rt_lumbar_preview.setStyleSheet(self.box_style)
        self.rt_lumbar_preview.setText("")
        self.rt_lumbar_preview.hide()

        # 薦椎預覽（右）
        self.rt_sacrum_preview = QtWidgets.QLabel(self.root)
        self.rt_sacrum_preview.setGeometry(690, 670, 200, 200)  # 在 frame6 下方
        self.rt_sacrum_preview.setStyleSheet(self.box_style)
        self.rt_sacrum_preview.setText("")
        self.rt_sacrum_preview.hide()

        # === 訊息主控台（多行，可自動捲動） ===
        self.msg_console = QtWidgets.QPlainTextEdit(self.root)
        self.msg_console.setGeometry(20, 650, 410, 140)
        self.msg_console.setReadOnly(True)
        self.msg_console.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.msg_console.setStyleSheet(
            "background-color: #FFFFFF; border: 2px solid black; color: #111; "
            "font-family: Consolas, 'Courier New', monospace; font-size: 13px; padding: 6px;"
        )
        self.msg_console.show()

    def button(self):
        font = QtGui.QFont(); font.setPointSize(16)

        # === 第一排：模式切換 ===
        self.btn_mode_stitch = QtWidgets.QPushButton("拼接模式", self.root)
        self.btn_mode_stitch.setGeometry(20, 30, 130, 36)
        self.btn_mode_stitch.setFont(font)
        self.btn_mode_stitch.setStyleSheet(self.active_button_style)
        self.btn_mode_stitch.clicked.connect(lambda: self.set_mode("stitch"))
        self.btn_mode_stitch.show()

        self.btn_mode_rt = QtWidgets.QPushButton("即時模式", self.root)
        self.btn_mode_rt.setGeometry(170, 30, 130, 36)
        self.btn_mode_rt.setFont(font)
        self.btn_mode_rt.setStyleSheet(self.button_style)
        self.btn_mode_rt.clicked.connect(lambda: self.set_mode("realtime"))
        self.btn_mode_rt.show()

        # 分隔符（兩種模式都可顯示）
        self.sep_label = QtWidgets.QLabel("|", self.root)
        self.sep_label.setGeometry(315, 30, 20, 36)
        self.sep_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sep_label.setStyleSheet("color:#333; font-size:20px;")
        self.sep_label.show()

        # === 即時子模式切換鈕 ===
        self.btn_rt_track = QtWidgets.QPushButton("追蹤", self.root)
        self.btn_rt_track.setGeometry(350, 30, 90, 36)
        self.btn_rt_track.setFont(font)
        self.btn_rt_track.setStyleSheet(self.button_style)
        self.btn_rt_track.clicked.connect(lambda: self.set_rt_submode("track"))
        self.btn_rt_track.hide()

        self.btn_rt_count = QtWidgets.QPushButton("計數", self.root)
        self.btn_rt_count.setGeometry(450, 30, 90, 36)
        self.btn_rt_count.setFont(font)
        self.btn_rt_count.setStyleSheet(self.button_style)
        self.btn_rt_count.clicked.connect(lambda: self.set_rt_submode("count"))
        self.btn_rt_count.hide()

        # —— 即時模式下「執行按鈕」：追蹤 / 計數 ——
        # 追蹤子模式：三顆按鈕
        self.btn_rt_track_start = QtWidgets.QPushButton('開始追蹤', self.root)
        self.btn_rt_track_start.setGeometry(480, 100, 120, 40)
        self.btn_rt_track_start.setFont(font)
        self.btn_rt_track_start.setStyleSheet(self.button_style)
        self.btn_rt_track_start.clicked.connect(self.start_track_thread)
        self.btn_rt_track_start.hide()

        self.btn_rt_track_stop = QtWidgets.QPushButton('停止追蹤', self.root)
        self.btn_rt_track_stop.setGeometry(480, 150, 120, 40)
        self.btn_rt_track_stop.setFont(font)
        self.btn_rt_track_stop.setStyleSheet(self.button_style)
        self.btn_rt_track_stop.clicked.connect(self.stop_track_thread)
        self.btn_rt_track_stop.hide()

        self.btn_rt_track_best1 = QtWidgets.QPushButton('判斷最佳腰椎', self.root)
        self.btn_rt_track_best1.setGeometry(630, 100, 150, 40)
        self.btn_rt_track_best1.setFont(font)
        self.btn_rt_track_best1.setStyleSheet(self.button_style)
        self.btn_rt_track_best1.clicked.connect(self.on_rt_judge_best)
        self.btn_rt_track_best1.hide()

        self.btn_rt_track_best2 = QtWidgets.QPushButton('判斷最佳薦椎', self.root)
        self.btn_rt_track_best2.setGeometry(630, 150, 150, 40)
        self.btn_rt_track_best2.setFont(font)
        self.btn_rt_track_best2.setStyleSheet(self.button_style)
        self.btn_rt_track_best2.clicked.connect(self.on_rt_judge_best1)
        self.btn_rt_track_best2.hide()

        self.btn_rt_track_stitch = QtWidgets.QPushButton('進行拼接', self.root)
        self.btn_rt_track_stitch.setGeometry(810, 100, 120, 40)
        self.btn_rt_track_stitch.setFont(font)
        self.btn_rt_track_stitch.setStyleSheet(self.button_style)
        self.btn_rt_track_stitch.clicked.connect(self.on_rt_stitch)
        self.btn_rt_track_stitch.hide()

        # 計數子模式：一顆按鈕（你原本就有）
        self.btn_rt_count_input = QtWidgets.QPushButton('開始計數', self.root)
        self.btn_rt_count_input.setGeometry(480, 100, 120, 40)
        self.btn_rt_count_input.setFont(font)
        self.btn_rt_count_input.setStyleSheet(self.button_style)
        self.btn_rt_count_input.clicked.connect(self.count_images)
        self.btn_rt_count_input.hide()

        self.btn_rt_count_stop = QtWidgets.QPushButton('停止計數', self.root)
        self.btn_rt_count_stop.setGeometry(480, 150, 120, 40)
        self.btn_rt_count_stop.setFont(font)
        self.btn_rt_count_stop.setStyleSheet(self.button_style)
        self.btn_rt_count_stop.clicked.connect(self.stop_count)
        self.btn_rt_count_stop.hide()


        # === 第二排：錄影 ===
        self.start_recording_btn = QtWidgets.QPushButton('開始錄影', self.root)
        self.start_recording_btn.setGeometry(20, 100, 120, 40)
        self.start_recording_btn.setFont(font)
        self.start_recording_btn.setStyleSheet(self.button_style)
        self.start_recording_btn.clicked.connect(self.start_recording)
        self.start_recording_btn.show()

        self.stop_recording_btn = QtWidgets.QPushButton('停止錄影', self.root)
        self.stop_recording_btn.setGeometry(270, 100, 120, 40)
        self.stop_recording_btn.setFont(font)
        self.stop_recording_btn.setStyleSheet(self.button_style)
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        self.stop_recording_btn.setDisabled(True)
        self.stop_recording_btn.show()

        # === 拼接模式按鈕 ===
        self.lumbar_input_btn = QtWidgets.QPushButton('腰椎輸入', self.root)
        self.lumbar_input_btn.setGeometry(480, 100, 120, 40)
        self.lumbar_input_btn.setFont(font); self.lumbar_input_btn.setStyleSheet(self.button_style)
        self.lumbar_input_btn.clicked.connect(self.select_and_display_image)
        self.lumbar_input_btn.show()

        self.lumbar_pretreatment_btn = QtWidgets.QPushButton('做前處理', self.root)
        self.lumbar_pretreatment_btn.setGeometry(480, 150, 120, 40)
        self.lumbar_pretreatment_btn.setFont(font); self.lumbar_pretreatment_btn.setStyleSheet(self.button_style)
        self.lumbar_pretreatment_btn.clicked.connect(self.handle_image_processing)
        self.lumbar_pretreatment_btn.show()

        self.sacral_input_btn = QtWidgets.QPushButton('薦椎輸入', self.root)
        self.sacral_input_btn.setGeometry(780, 100, 120, 40)
        self.sacral_input_btn.setFont(font); self.sacral_input_btn.setStyleSheet(self.button_style)
        self.sacral_input_btn.clicked.connect(self.select_and_display_image1)
        self.sacral_input_btn.show()

        self.sacral_pretreatment_btn = QtWidgets.QPushButton('做前處理', self.root)
        self.sacral_pretreatment_btn.setGeometry(780, 150, 120, 40)
        self.sacral_pretreatment_btn.setFont(font); self.sacral_pretreatment_btn.setStyleSheet(self.button_style)
        self.sacral_pretreatment_btn.clicked.connect(self.handle_image_processing1)
        self.sacral_pretreatment_btn.show()

        self.segmentation_btn = QtWidgets.QPushButton('分割', self.root)
        self.segmentation_btn.setGeometry(1080, 100, 120, 40)
        self.segmentation_btn.setFont(font); self.segmentation_btn.setStyleSheet(self.button_style)
        self.segmentation_btn.clicked.connect(self.unet_segmentation)  # ← 確認方法名稱
        self.segmentation_btn.show()

        self.contour_btn = QtWidgets.QPushButton('輪廓', self.root)
        self.contour_btn.setGeometry(1080, 150, 120, 40)
        self.contour_btn.setFont(font); self.contour_btn.setStyleSheet(self.button_style)
        self.contour_btn.clicked.connect(self.run_segmentation_pipeline)
        self.contour_btn.show()

        self.splicing_btn = QtWidgets.QPushButton('拼接', self.root)
        self.splicing_btn.setGeometry(1230, 100, 120, 40)
        self.splicing_btn.setFont(font); self.splicing_btn.setStyleSheet(self.button_style)
        self.splicing_btn.clicked.connect(self.stitch_images)
        self.splicing_btn.show()

        # —— 清空輸出（依當前模式）——
        self.btn_clear_outputs = QtWidgets.QPushButton("清空輸出", self.root)
        self.btn_clear_outputs.setGeometry(550, 30, 120, 36)
        self.btn_clear_outputs.setFont(font)
        self.btn_clear_outputs.setStyleSheet(self.button_style)
        self.btn_clear_outputs.clicked.connect(self.clear_current_mode_outputs)
        self.btn_clear_outputs.show()

    def label(self):
        # 文字
        font = QtGui.QFont(); font.setPointSize(16)
        small = QtGui.QFont(); small.setPointSize(13)

        # 錄影狀態與時間
        self.record_label = QtWidgets.QLabel(self.root)
        self.record_label.setText('Ready')
        self.record_label.move(150, 100)
        self.record_label.setFont(font)
        self.record_label.setStyleSheet("color: black; background: transparent;")
        self.record_label.show()

        self.recording_time_label = QtWidgets.QLabel('00:00:00', self.root)
        self.recording_time_label.move(150, 150)
        self.recording_time_label.setFont(font)
        self.recording_time_label.setStyleSheet("color: black; background: transparent;")
        self.recording_time_label.show()

        # === 框上方小標題（拼接模式頁面）===
        self.cap_lumbar = QtWidgets.QLabel("腰椎畫面", self.root)
        self.cap_lumbar.setStyleSheet(self.caption_style)
        self.cap_lumbar.move(self.frame1.x(), self.frame1.y()-28)
        self.cap_lumbar.show()

        self.cap_sacrum = QtWidgets.QLabel("薦椎畫面", self.root)
        self.cap_sacrum.setStyleSheet(self.caption_style)
        self.cap_sacrum.move(self.frame2.x(), self.frame2.y()-28)
        self.cap_sacrum.show()

        self.cap_stitch2d = QtWidgets.QLabel("影像拼接", self.root)
        self.cap_stitch2d.setStyleSheet(self.caption_style)
        self.cap_stitch2d.move(self.frame3.x(), self.frame3.y()-28)
        self.cap_stitch2d.show()

        self.cap_lumbar_unet = QtWidgets.QLabel("腰椎Unet", self.root)
        self.cap_lumbar_unet.setStyleSheet(self.caption_style)
        self.cap_lumbar_unet.move(self.frame4.x(), self.frame4.y()-28)
        self.cap_lumbar_unet.show()

        self.cap_sacrum_unet = QtWidgets.QLabel("薦椎Unet", self.root)
        self.cap_sacrum_unet.setStyleSheet(self.caption_style)
        self.cap_sacrum_unet.move(self.frame5.x(), self.frame5.y()-28)
        self.cap_sacrum_unet.show()

        # === 框上方小標題（即時模式頁面）===
        self.cap_live = QtWidgets.QLabel("即時錄影", self.root)
        self.cap_live.setStyleSheet(self.caption_style)
        self.cap_live.move(self.ultra_frame.x(), self.ultra_frame.y()-28)
        self.cap_live.show()

        self.cap_rt = QtWidgets.QLabel("即時處理", self.root)
        self.cap_rt.setStyleSheet(self.caption_style)
        self.cap_rt.move(self.frame6.x(), self.frame6.y()-28)
        self.cap_rt.hide()

        self.cap_rt_stitch = QtWidgets.QLabel("即時拼接", self.root)
        self.cap_rt_stitch.setStyleSheet(self.caption_style)
        self.cap_rt_stitch.move(self.frame7.x(), self.frame7.y()-28)
        self.cap_rt_stitch.hide()

        # 小預覽標題（即時模式）
        self.cap_rt_lumbar_pick = QtWidgets.QLabel("腰椎", self.root)
        self.cap_rt_lumbar_pick.setStyleSheet(self.caption_style)
        self.cap_rt_lumbar_pick.move(self.rt_lumbar_preview.x(), self.rt_lumbar_preview.y() - 28)
        self.cap_rt_lumbar_pick.hide()

        self.cap_rt_sacrum_pick = QtWidgets.QLabel("薦椎", self.root)
        self.cap_rt_sacrum_pick.setStyleSheet(self.caption_style)
        self.cap_rt_sacrum_pick.move(self.rt_sacrum_preview.x(), self.rt_sacrum_preview.y() - 28)
        self.cap_rt_sacrum_pick.hide()

    # ========= 清空輸出（依目前模式） =========
    def clear_current_mode_outputs(self):
        """
        依 self.mode 清理輸出資料夾內容：
        - stitch: 清 data/stitching/{selected, preprocessed, unet, contour, meta, stitched, stitched/error}
        - realtime: 清 data/realtime/{track, count}
        只清『內容』，會保留資料夾本身。
        """
        if self.mode not in ("stitch", "realtime"):
            QtWidgets.QMessageBox.information(self.root, "清空輸出", "目前模式未知，無法清理。")
            return

        if self.mode == "stitch":
            targets = [
                self._p("stitching", "selected"),
                self._p("stitching", "preprocessed"),
                self._p("stitching", "unet"),
                self._p("stitching", "contour"),
                self._p("stitching", "meta"),
                self._p("stitching", "stitched"),
                self._p("stitching", "stitched", "error"),
            ]
            mode_text = "拼接模式輸出（stitching）"
        else:
            # realtime：同時清 track 與 count 和 _live
            targets = [
                self._p("realtime", "track"),
                self._p("realtime", "count"),
                self._p("realtime", "_live"),
            ]
            mode_text = "即時模式輸出（realtime/track + realtime/count + realtime/_live）"


        # 確認對話框
        msg = f"確定要清空 {mode_text} 的所有檔案嗎？\n此動作不可復原。"
        reply = QtWidgets.QMessageBox.question(
            self.root, "清空輸出", msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        total_files, total_dirs = 0, 0
        for d in targets:
            f_cnt, d_cnt = self._clear_dir_contents(d)
            total_files += f_cnt
            total_dirs  += d_cnt

        # 清空畫面上的預覽（避免殘留舊圖）
        try:
            if self.mode == "stitch":
                for lab in [self.frame1, self.frame2, self.frame3, self.frame4, self.frame5]:
                    lab.clear()
                    lab.setText("")
            else:
                for lab in [self.frame6, self.frame7, self.rt_lumbar_preview, self.rt_sacrum_preview]:
                    lab.clear()
                    lab.setText("")
        except Exception:
            pass

        info = f"已清除檔案 {total_files} 個、子資料夾 {total_dirs} 個。"
        self.log(info)
        QtWidgets.QMessageBox.information(self.root, "清空完成", info)

    def _clear_dir_contents(self, dir_path: str):
        """
        刪除 dir_path 下『所有內容』，但保留資料夾本身。
        回傳 (刪除檔案數, 刪除子資料夾數)。
        """
        files_deleted, dirs_deleted = 0, 0
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                return (0, 0)

            for name in os.listdir(dir_path):
                p = os.path.join(dir_path, name)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.remove(p)
                        files_deleted += 1
                    elif os.path.isdir(p):
                        sh.rmtree(p)
                        dirs_deleted += 1
                except Exception as e:
                    self.log(f"刪除失敗：{p} -> {e}")
        except Exception as e:
            self.log(f"清理資料夾失敗：{dir_path} -> {e}")
        finally:
            # 確保資料夾存在
            os.makedirs(dir_path, exist_ok=True)

        return (files_deleted, dirs_deleted)

    def imaging(self):
        """
        擷取 Ultrasound 畫面並：
        - 即時顯示到 GUI (ultra_frame)
        - 若啟用 rt_use_ram：同時推入 frame_queue、frame_store (給 YOLO/追蹤)
        """
        import time, os, cv2, traceback
        import numpy as np
        from PIL import ImageGrab

        try:
            # --- 初始化路徑 ---
            if not hasattr(self, "DATA_ROOT"):
                self._init_paths()

            # --- 擷取畫面 ---
            img_rgb = ImageGrab.grab(bbox=(self.x, self.y, self.x + self.width, self.y + self.height))
            img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

            # --- 寫出即時顯示檔 ---
            live_dir = self._p("realtime", "_live")
            os.makedirs(live_dir, exist_ok=True)
            out_path = os.path.join(live_dir, "live_ULTRASOUND.jpg")
            cv2.imwrite(out_path, img_bgr)

            # --- 顯示到 ultra_frame ---
            if hasattr(self, "_log_emitter"):
                self._log_emitter.show_image.emit("ultra_frame", out_path)

            # --- 保留影像到記憶體 (RAM 模式) ---
            self.img_array = img_bgr

            if getattr(self, "rt_use_ram", False):
                # 初始化暫存結構
                if not hasattr(self, "frame_queue"):
                    from collections import deque
                    self.frame_queue = deque(maxlen=512)
                if not hasattr(self, "frame_store"):
                    self.frame_store = {}
                if not hasattr(self, "frame_store_max"):
                    self.frame_store_max = 256

                fid = getattr(self, "next_frame_id", 0)
                self.frame_queue.append((fid, img_bgr.copy()))
                self.frame_store[fid] = img_bgr.copy()

                # 控制記憶體大小，移除最舊影像
                if len(self.frame_store) > self.frame_store_max:
                    oldest = min(self.frame_store.keys())
                    self.frame_store.pop(oldest, None)

                # 遞增影格 ID
                self.next_frame_id = fid + 1

                # 每 10 幀 log 一次
                #if fid % 10 == 0:
                #    self.log(f"[CAP] push fid={fid} queue={len(self.frame_queue)} store={len(self.frame_store)}")

        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"即時擷取失敗：{e}\n{tb}", level="error")

        # 控制擷取頻率
        time.sleep(0.1)  # 約 10 fps


    def loop(self):
        while self.save:
            self.imaging()

    def start(self):
        self.T = QtCore.QThread()
        self.T.run = self.loop
        self.T.start()
    
    def start_recording(self):
        self.recording = True
        self.stop_recording_event.clear()  # ← 改這顆
        self.start_recording_btn.setEnabled(False)
        self.stop_recording_btn.setEnabled(True)
        self.record_label.setText("錄製中...")
        self.record_timer.start(1000)  # 每秒更新一次時間

        # 啟動錄製的執行緒
        self.record_thread = QtCore.QThread()
        self.record_thread.run = self.record_video
        self.record_thread.start()

    def stop_recording(self):
        self.recording = False
        self.stop_recording_event.set()    # ← 改這顆
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.record_label.setText("錄製結束")
        self.record_timer.stop()
        self.time = QTime(0, 0, 0)
        self.recording_time_label.setText(self.time.toString("hh:mm:ss"))

    def record_video(self):
        # 準備當天錄影的根目錄
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y%m%d")
        main_folder = self._p("recording", date_folder)  # 會自動建立

        # 取得新的 session 編號
        existing = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f)) and f.isdigit()]
        new_folder_num = (max(map(int, existing)) + 1) if existing else 1

        # 建立本次輸出目錄
        output_folder = self._p("recording", date_folder, str(new_folder_num))  # 自動建立

        # 錄影主迴圈
        frame_interval = 1 / 30.0           # 30 fps
        last_frame_ts = time.time()
        frame_count = 0

        # [MOD] 相容：若你已分離 stop_recording_event 就用它，否則回退用 stop_event
        stop_flag = getattr(self, "stop_recording_event", None)
        if stop_flag is None:
            stop_flag = getattr(self, "stop_event", None)

        while self.recording and not (stop_flag.is_set() if stop_flag is not None else False):
            now_ts = time.time()
            if (now_ts - last_frame_ts) >= frame_interval:
                try:
                    # 擷取螢幕（可能因權限/多螢幕出錯，包 try）
                    img_rgb = ImageGrab.grab(bbox=(self.x, self.y, self.x + self.width, self.y + self.height))
                    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

                    # 存檔
                    frame_filename = f"frame_{frame_count:06d}.png"
                    out_path = os.path.join(output_folder, frame_filename)
                    ok = cv2.imwrite(out_path, img_bgr)
                    if not ok:
                        raise RuntimeError("cv2.imwrite 失敗")

                    # [MOD] 透過訊號在主執行緒顯示當前畫面（避免子執行緒直接碰 UI）
                    if hasattr(self, "_log_emitter"):
                        self._log_emitter.show_image.emit("ultra_frame", out_path)

                    last_frame_ts = now_ts
                    frame_count += 1

                except Exception as e:
                    # [MOD] 擷取失敗時記錄並略過該幀，避免整個錄影中斷
                    self.log(f"錄影擷取/存檔失敗：{e}")
                    # 給系統一點緩衝時間，避免密集重試
                    time.sleep(0.02)

            else:
                # 補點 sleep 減少 busy wait
                time.sleep(0.001)

    def update_time(self):
        """
        錄影時間計時：每秒被 QTimer 觸發一次。
        - 只在 self.recording 為 True 時累加（避免停掉後還繼續跑）
        - 將時間更新到 self.recording_time_label
        """
        try:
            if getattr(self, "recording", False):
                # 累加 1 秒
                self.time = self.time.addSecs(1)
                # 更新 UI
                if hasattr(self, "recording_time_label") and self.recording_time_label is not None:
                    self.recording_time_label.setText(self.time.toString("hh:mm:ss"))
        except Exception as e:
            # 發生任何錯誤不要讓計時器崩潰
            self.log(f"update_time 發生例外：{e}")

    @staticmethod
    def condition_function(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return brightness > 100  # 例：亮度 > 100 才保留
    
    def filter_images(self, input_folder, output_folder):

        # 創建輸出資料夾
        os.makedirs(output_folder, exist_ok=True)

        # 獲取資料夾中的所有圖片檔案
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 遍歷所有圖片
        for image_file in image_files:
            input_path = os.path.join(input_folder, image_file)
            
            # 使用 OpenCV 讀取圖片
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue
            
            # 使用條件函數判斷是否符合要求
            if self.condition_function(image):
                # 符合條件，保存圖片到輸出資料夾
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, image)

    def run_filter_images(self):
        # 取得最新錄影 session
        latest_dir, session_folder = self._latest_recording_session()
        if latest_dir is None:
            self.log(f"找不到錄影資料。")
            return

        input_folder = latest_dir
        output_folder = self._p("recording", session_folder.split("_")[0], session_folder.split("_")[1], "filtered")
        # 等價於：data/recording/YYYYMMDD/N/filtered

        self.filter_images(input_folder, output_folder)
        self.log(f"已篩選：{input_folder} → {output_folder}")

    def preprocess_images_and_labels(self, folder_path, target_size):
        if not os.path.exists(folder_path):
            return

        # 處理影像
        image_files = glob.glob(os.path.join(folder_path, '*.[jp][pn]g'))  # 找出資料夾中是 jpg, png 的檔案
        for image_file in image_files:
            # 使用 TensorFlow 讀取和處理影像
            image = tf.io.read_file(image_file)
            image = tf.image.decode_png(image, channels=3)  # 灰階影像通道為 1
            image = tf.image.convert_image_dtype(image, tf.float32)  # 轉為 float32 (方便後續處理)
            
            resized_image = tf.image.resize_with_pad(
                image,
                target_size[0],
                target_size[1],
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            # 轉換為 uint8 格式並保存
            resized_image_uint8 = tf.image.convert_image_dtype(resized_image, tf.uint8)
            output_path = os.path.join(folder_path, os.path.basename(image_file))
            resized_image_png = tf.io.encode_png(resized_image_uint8)
            tf.io.write_file(output_path, resized_image_png)

        # 處理標籤
        label_files = glob.glob(os.path.join(folder_path, '*.npy'))  # 找出資料夾中 .npy 的檔案
        for label_file in label_files:
            # 使用 NumPy 讀取和處理標籤
            label = np.load(label_file)
            label = label[..., np.newaxis]  # 添加維度以符合 resize 函數要求
            
            resized_label = tf.image.resize_with_pad(
                label,
                target_size[0],
                target_size[1],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            resized_label = tf.squeeze(resized_label).numpy()  # 移除大小為 1 的維度
            
            output_path = os.path.join(folder_path, os.path.basename(label_file))
            np.save(output_path, resized_label)

    # 把 QPixmap 塞進指定框，維持等比縮放
    def _show_pixmap(self, target_label: QtWidgets.QLabel, path: str):
        pix = QtGui.QPixmap(path)
        target_label.setPixmap(
            pix.scaled(
                target_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
        )
        target_label.setText("")
        target_label.setScaledContents(False)  # 保持等比，不強制拉伸

    def stitch(self):
    # 確保 frame1 和 frame3 的圖像已擷取
        if self.frame1.pixmap() is None or self.frame3.pixmap() is None:
            print("Error: Images from frame1 or frame3 are missing!")
            return

        # 將 QPixmap 轉換為 QImage
        image1 = self.frame1.pixmap().toImage()
        image2 = self.frame3.pixmap().toImage()

        # 將 QImage 轉換為 numpy 陣列
        ptr1 = image1.bits()
        ptr1.setsize(image1.sizeInBytes())
        img1 = np.array(ptr1).reshape(image1.height(), image1.width(), 4)[:, :, :3]  # 去掉 alpha 通道

        ptr2 = image2.bits()
        ptr2.setsize(image2.sizeInBytes())
        img2 = np.array(ptr2).reshape(image2.height(), image2.width(), 4)[:, :, :3]  # 去掉 alpha 通道

        # 使用 OpenCV 的 SIFT 進行拼接
        try:
            sift = cv2.SIFT_create()

            # 檢測並計算特徵點和描述子
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

            # 使用 FLANN 匹配特徵點
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)

            # 過濾匹配點
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                print("Error: Not enough matches for stitching!")
                return

            # 獲取匹配的點
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 計算透視變換矩陣
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            # 將第二張圖片進行透視變換
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            result = cv2.warpPerspective(img2, M, (w1 + w2, max(h1, h2)))
            result[0:h1, 0:w1] = img1

            # 將拼接結果轉換為 QPixmap
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            result_qt = QtGui.QImage(result.data, result.shape[1], result.shape[0],
                                    result.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
            result_pixmap = QtGui.QPixmap.fromImage(result_qt)

            # 顯示拼接後的圖像在 frame2
            self.frame2.setPixmap(result_pixmap)
            self.frame2.setScaledContents(True)  # 確保圖像適配框架大小

        except Exception as e:
            print(f"Error during stitching: {e}")



    def select_and_display_image(self):
    # 選擇檔案的對話框
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.root,
            "選擇圖片",  # 對話框標題
            "",  # 起始目錄，留空為當前目錄
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)",  # 檔案過濾條件
        )

        if file_path and os.path.isfile(file_path):
            save_dir = "saved_images"
            os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在

            # 計算新的檔案名稱編號
            existing_files = [f for f in os.listdir(save_dir) if f.startswith("Lumbar_") and f.endswith("_before.jpg")]
            max_index = 0
            for f in existing_files:
                try:
                    index = int(f.split("_")[1])
                    max_index = max(max_index, index)
                except ValueError:
                    continue
            
            new_file_name = f"Lumbar_{max_index + 1}_before.jpg"
            save_path = os.path.join(save_dir, new_file_name)
            
            sh.copy(file_path, save_path)  # 複製檔案到指定資料夾

            # 顯示圖片
            pixmap = QtGui.QPixmap(save_path)
            self.frame1.setPixmap(
                pixmap.scaled(
                    self.frame1.size(),
                    aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.frame1.setText("")  # 移除預設文字
        else:
            print("未選擇檔案或無效的檔案")



    def select_and_display_image1(self):
        # 選擇檔案的對話框
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.root,
            "選擇圖片",  # 對話框標題
            "",  # 起始目錄，留空為當前目錄
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)",  # 檔案過濾條件
        )

        if file_path and os.path.isfile(file_path):
            save_dir = "saved_images"
            os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在

            # 計算新的檔案名稱編號
            existing_files = [f for f in os.listdir(save_dir) if f.startswith("Sacrum_") and f.endswith("_before.jpg")]
            max_index = 0
            for f in existing_files:
                try:
                    index = int(f.split("_")[1])
                    max_index = max(max_index, index)
                except ValueError:
                    continue
            
            new_file_name = f"Sacrum_{max_index + 1}_before.jpg"
            save_path = os.path.join(save_dir, new_file_name)
            
            sh.copy(file_path, save_path)  # 複製檔案到指定資料夾

            # 顯示圖片
            pixmap = QtGui.QPixmap(save_path)
            self.frame2.setPixmap(
                pixmap.scaled(
                    self.frame2.size(),
                    aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.frame2.setText("")  # 移除預設文字
        else:
            print("未選擇檔案或無效的檔案")
    

    def handle_image_processing(self):
        save_dir = "saved_images"
        target_size = (512, 512)
        
        # 找到最新的 Lumbar_X_before 圖片
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("Lumbar_") and f.endswith("_before.jpg")]
        max_index = 0
        latest_file = None
        
        for f in existing_files:
            try:
                index = int(f.split("_")[1])
                if index > max_index:
                    max_index = index
                    latest_file = f
            except ValueError:
                continue
        
        if latest_file is None:
            print("未找到有效的圖片進行處理")
            return
        
        input_path = os.path.join(save_dir, latest_file)
        output_filename = f"Lumbar_{max_index}_after.jpg"
        output_path = os.path.join(save_dir, output_filename)
        
        with Image.open(input_path) as img:
            x, y = 167, 59  # 剪裁座標
            width, height = 690, 551
            right, bottom = x + width, y + height
            cropped_img = img.crop((x, y, right, bottom))
            cropped_img.save(output_path)
        
        image = tf.io.read_file(output_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        resized_image = tf.image.resize_with_pad(
            image, target_size[0], target_size[1], method=tf.image.ResizeMethod.BILINEAR
        )
        resized_image_uint8 = tf.image.convert_image_dtype(resized_image, tf.uint8)
        resized_image_png = tf.io.encode_png(resized_image_uint8)
        tf.io.write_file(output_path, resized_image_png)
        
        pixmap = QtGui.QPixmap(output_path)
        self.frame1.setPixmap(pixmap)
        self.frame1.setScaledContents(True)

    def handle_image_processing1(self):
        save_dir = "saved_images"
        target_size = (512, 512)
        
        # 找到最新的 Sacrum_X_before 圖片
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("Sacrum_") and f.endswith("_before.jpg")]
        max_index = 0
        latest_file = None
        
        for f in existing_files:
            try:
                index = int(f.split("_")[1])
                if index > max_index:
                    max_index = index
                    latest_file = f
            except ValueError:
                continue
        
        if latest_file is None:
            print("未找到有效的圖片進行處理")
            return
        
        input_path = os.path.join(save_dir, latest_file)
        output_filename = f"Sacrum_{max_index}_after.jpg"
        output_path = os.path.join(save_dir, output_filename)
        
        with Image.open(input_path) as img:
            x, y = 167, 59  # 剪裁座標
            width, height = 690, 551
            right, bottom = x + width, y + height
            cropped_img = img.crop((x, y, right, bottom))
            cropped_img.save(output_path)
        
        image = tf.io.read_file(output_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        resized_image = tf.image.resize_with_pad(
            image, target_size[0], target_size[1], method=tf.image.ResizeMethod.BILINEAR
        )
        resized_image_uint8 = tf.image.convert_image_dtype(resized_image, tf.uint8)
        resized_image_png = tf.io.encode_png(resized_image_uint8)
        tf.io.write_file(output_path, resized_image_png)
        
        pixmap = QtGui.QPixmap(output_path)
        self.frame2.setPixmap(pixmap)
        self.frame2.setScaledContents(True)


    def handle_image_processing2(self, target_size=(512, 512), crop_coords=(0, 54, 512, 404)):
        """
        這個函數結合了裁切和縮放的功能，先裁切圖片，再進行填充縮放，並儲存處理後的結果。
        source_dir: 圖片的來源資料夾
        target_dir: 儲存處理後圖片的目標資料夾
        target_size: 目標尺寸 (height, width)
        crop_coords: 裁切範圍 (x, y, width, height)，預設為 (0, 54, 512, 404)
        """
        
        # 設定圖片來源和輸出資料夾路徑
        image_dir = 'C:/Users/Janice/Downloads/DDH_超象/DDH_超象/saved_images'
        image_output_dir = 'C:/Users/Janice/Downloads/DDH_超象/DDH_超象/saved_images_output'
        
        # 確保目標資料夾存在
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
        
        # 找出所有的圖片檔案
        image_files = glob.glob(os.path.join(image_dir, '*.[jp][pn]g'))
        
        for image_file in image_files:
            # 使用 PIL 進行裁切
            with Image.open(image_file) as img:
                x, y, width, height = crop_coords
                right = x + width
                bottom = y + height
                cropped_img = img.crop((x, y, right, bottom))
                
                # 保存裁切後的圖像到新的資料夾
                cropped_img_path = os.path.join(image_output_dir, os.path.basename(image_file))
                cropped_img.save(cropped_img_path)
            
            # 使用 TensorFlow 進行填充和縮放
            image = tf.io.read_file(cropped_img_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            
            # 填充並縮放圖片
            print(target_size, type(target_size))
            resized_image = tf.image.resize_with_pad(
                image, target_size[0], target_size[1], method=tf.image.ResizeMethod.BILINEAR
            )
            
            # 轉回 uint8 類型並保存
            resized_image_uint8 = tf.image.convert_image_dtype(resized_image, tf.uint8)
            output_path = os.path.join(image_output_dir, os.path.basename(image_file))
            resized_image_png = tf.io.encode_png(resized_image_uint8)
            tf.io.write_file(output_path, resized_image_png)
            
            # 在 GUI 顯示處理後的圖像
            pixmap = QtGui.QPixmap(output_path)
            self.frame1.setPixmap(pixmap)
            self.frame1.setScaledContents(True)

    #==============分割==================
    def dice_loss_with_clip(self, y_true, y_pred):
        """
        自定義 Dice Loss，並限制其值介於 0 和 1 之間
        """
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1]))(y_true, y_pred)
        return tf.keras.backend.clip(dice_loss, 0, 1)

    def dice_loss_plus_1focal_loss(self, y_true, y_pred):
        """
        Dice Loss (限制最大值為 1) + Categorical Focal Loss
        """
        focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
        return self.dice_loss_with_clip(y_true, y_pred) + focal_loss

    def load_model_with_custom_objects(self, model_path):
        """
        只在初始化時載入模型，不在 segmentation 階段重複載入
        """
        custom_objects = {
            'dice_loss_plus_1focal_loss': self.dice_loss_plus_1focal_loss,
            'iou_score': sm.metrics.IOUScore(),
            'f1-score': sm.metrics.FScore(),
        }

        try:
            model = load_model(model_path, custom_objects=custom_objects)
            print(f"{model_path} 模型載入成功")
            return model
        except Exception as e:
            print(f"{model_path} 模型載入失敗: {e}")
            return None

    def get_latest_file(self, save_dir, prefix, middle):
        """
        回傳 prefix_X_middle.xxx 的最新檔案
        例如：Lumbar_4_after.png、Sacrum_12_after.jpg、Lumbar_7_contour.json
        不限制副檔名，只要長得像 prefix_xxx_middle.*
        """
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".json")  # 允許多種格式

        existing_files = []
        for f in os.listdir(save_dir):
            if f.startswith(prefix + "_") and middle in f and f.lower().endswith(valid_ext):
                existing_files.append(f)

        latest_file = None
        max_index = -1

        for f in existing_files:
            try:
                index = int(f.split("_")[1])
                if index > max_index:
                    max_index = index
                    latest_file = f
            except:
                continue

        return latest_file


    def process_segmentation(self, model, save_dir, prefix, suffix, frame):
        """
        使用已載入的模型進行影像分割，並移除面積小於 250 的雜訊輪廓。
        """
        target_size = (512, 512)

        # 找到最新的圖片
        latest_file = self.get_latest_file(save_dir, prefix, suffix)

        if not latest_file:
            print(f"{prefix} 未找到有效的圖片進行分割處理")
            return

        input_path = os.path.join(save_dir, latest_file)
        output_filename = f"{prefix}_{latest_file.split('_')[1]}_unet.png"
        output_path = os.path.join(save_dir, output_filename)

        # 讀取圖像並進行預處理
        img = load_img(input_path, target_size=target_size, color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # 歸一化
        img_array = np.concatenate([img_array] * 3, axis=-1)  # (512, 512, 3)
        img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度

        # 進行模型預測
        prediction = model.predict(img_array)
        prediction_image = np.squeeze(prediction)  # 去掉 batch 維度

        # 轉換為分類索引
        prediction_classes = np.argmax(prediction_image, axis=-1)  # 形狀為 (512, 512)
        class_colors = {
            0: (0, 0, 0),         # 黑色
            1: (255, 0, 0),       # 紅色
            2: (0, 255, 0),       # 綠色
            3: (0, 0, 255),       # 藍色
        }

        height, width = prediction_classes.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, color in class_colors.items():
            mask = (prediction_classes == class_idx)
            color_image[mask] = color

        # **轉換為 OpenCV 格式**
        filtered_image = np.array(color_image, dtype=np.uint8)

        # **轉換為灰階並尋找輪廓**
        gray = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # **移除面積小於 250 的輪廓**
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 250:
                cv2.drawContours(filtered_image, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

        # **儲存結果**
        Image.fromarray(filtered_image).save(output_path)

        # 顯示結果
        pixmap = QtGui.QPixmap(output_path)
        frame.setPixmap(pixmap)
        frame.setScaledContents(True)

    def unet_segmentation(self):
        """
        進行 U-Net 分割（但不重複載入模型）
        """
        save_dir = "saved_images"

        # 直接使用已載入的模型
        if self.lumbar_model:
            self.process_segmentation(self.lumbar_model, save_dir, "Lumbar", "_after.jpg", self.frame4)

        if self.sacrum_model:
            self.process_segmentation(self.sacrum_model, save_dir, "Sacrum", "_after.jpg", self.frame5)
        
    #==============輪廓==================
    def run_segmentation_pipeline(self):
        """
        1. 找出最新的 Lumbar_X_unet.png 和 Sacrum_X_unet.png
        2. 針對 Lumbar & Sacrum 影像使用不同方式標註 L5
        3. 儲存最終結果 & 生成 JSON
        4. 直接顯示影像於 frame4 & frame5
        """
        save_dir = "saved_images"
        
        # 取得最新影像
        lumbar_file = self.find_latest_lumbar_image()
        sacrum_file = self.find_latest_sacrum_image()
        
        if not lumbar_file:
            print("未找到 Lumbar_X_unet.png")
            return

        if not sacrum_file:
            print("未找到 Sacrum_X_unet.png")
            return

        lumbar_input_path = os.path.join(self.save_dir, lumbar_file)
        sacrum_input_path = os.path.join(self.save_dir, sacrum_file)

        lumbar_index = lumbar_file.split("_")[1]  # 取得 X
        sacrum_index = sacrum_file.split("_")[1]  # 取得 X

        lumbar_output_path = os.path.join(save_dir, f"Lumbar_{lumbar_index}_contour.png")
        sacrum_output_path = os.path.join(save_dir, f"Sacrum_{sacrum_index}_contour.png")

        lumbar_json_path = os.path.join(save_dir, f"Lumbar_{lumbar_index}_contour.json")
        sacrum_json_path = os.path.join(save_dir, f"Sacrum_{sacrum_index}_contour.json")

        # 定義 Lumbar 色彩範圍
        green_lower, green_upper = np.array([40, 40, 40]), np.array([80, 255, 255])
        red_lower, red_upper = np.array([0, 50, 50]), np.array([10, 255, 255])
        lumbar_color_ranges = [(green_lower, green_upper), (red_lower, red_upper)]
        lumbar_labels_colors = [("conus", (0, 255, 0)), ("bone", (0, 0, 255))]

        # 處理 Lumbar 影像 並取得 L5 座標
        lumbar_data = self.find_and_draw_contours(lumbar_input_path, lumbar_output_path, lumbar_color_ranges, lumbar_labels_colors)

        # 儲存 Lumbar L5 座標至 JSON
        with open(lumbar_json_path, "w") as json_file:
            json.dump(lumbar_data, json_file, indent=4)

        # **直接顯示影像於 frame4**
        self.frame4.setPixmap(QtGui.QPixmap(lumbar_output_path))
        self.frame4.setScaledContents(True)

        # 處理 Sacrum 影像 並取得 L5 座標
        sacrum_data = self.process_image(sacrum_input_path, sacrum_output_path)

        # 儲存 Sacrum L5 座標至 JSON
        with open(sacrum_json_path, "w") as json_file:
            json.dump(sacrum_data, json_file, indent=4)

        # **直接顯示影像於 frame5**
        self.frame5.setPixmap(QtGui.QPixmap(sacrum_output_path))
        self.frame5.setScaledContents(True)

    def find_latest_lumbar_image(self):
        """從 saved_images 資料夾找出最新的 Lumbar_X_unet.png"""
        files = [f for f in os.listdir(self.save_dir) if f.startswith("Lumbar_") and f.endswith("_unet.png")]
        if not files:
            return None
        return max(files, key=lambda f: int(f.split("_")[1]))

    def find_latest_sacrum_image(self):
        """從 saved_images 資料夾找出最新的 Sacrum_X_unet.png"""
        files = [f for f in os.listdir(self.save_dir) if f.startswith("Sacrum_") and f.endswith("_unet.png")]
        if not files:
            return None
        return max(files, key=lambda f: int(f.split("_")[1]))

    def find_and_draw_contours(self, input_image_path, output_image_path, color_ranges, labels_colors):
        """
        處理 Lumbar 影像，標記 L5 -> L4 -> L3 -> L2
        若斜率變化明顯，排除 S1；若無明顯變化，則不刪除 S1
        """
        img = cv2.imread(input_image_path)
        contour_img = np.copy(img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        centers_coord = {"L2": None, "L3": None, "L4": None, "L5": None}

        for (lower, upper), (label, color) in zip(color_ranges, labels_colors):
            mask = cv2.inRange(hsv_img, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            centers = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if label == "bone" and area >= 250:  # 過濾掉面積過小的輪廓
                    valid_contours.append(cnt)
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        centers.append((cx, cy, cnt))

            # **確保有足夠的標記點**
            centers.sort(key=lambda x: x[0])  # 按 x 軸由左到右排序

            if label == "bone" and len(centers) < 4:
                print("腰椎圖片錯誤：節點數不足")
                warning_text = "wrong picture"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_thickness = 2

                text_size = cv2.getTextSize(warning_text, font, 1, font_thickness)[0]
                text_x = max(10, (img.shape[1] - text_size[0]) // 2)
                text_y = max(50, 50)  # 避免太靠邊界

                # 在圖片上顯示錯誤訊息
                cv2.putText(contour_img, warning_text, (text_x, text_y), font, 1, (0, 0, 255), font_thickness)
                cv2.imwrite(output_image_path, contour_img)
                return {"error": "腰椎圖片錯誤"}  # 回傳錯誤訊息
            
            if label == "bone" and len(centers) >= 4:  # 確保至少 4 個點 (L5, L4, L3, L2)
                # **步驟 1: 計算斜率變化**
                slopes = []
                for i in range(len(centers) - 1):
                    x1, y1 = centers[i][:2]
                    x2, y2 = centers[i + 1][:2]
                    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")
                    slopes.append(slope)

                # 找出斜率變化最大的地方 (S1)
                max_change = 0
                index_of_max_change = 0
                previous_slope = slopes[0]

                for i in range(1, len(slopes)):
                    change = abs(slopes[i] - previous_slope)
                    if change > max_change:
                        max_change = change
                        index_of_max_change = i
                    previous_slope = slopes[i]

                # **步驟 2: 是否需要標記 S1**
                slope_threshold = 0.4  # 斜率變化閾值 (可以根據數據調整)
                has_s1 = max_change > slope_threshold

                if has_s1:
                    print(f"檢測到 S1，將其標記 (斜率變化: {max_change})")

                # **步驟 3: 重新標記 S1 -> L5 -> L4 -> L3 -> L2**
                if has_s1 and len(centers) >= 5:
                    selected_centers = centers[-5:]  # 取最後 5 個點 (S1 → L5 → L4 → L3 → L2)
                else:
                    selected_centers = centers[-4:]  # 若無 S1 或點數不足，則標記 L5 → L2

                for i, (cx, cy, cnt) in enumerate(reversed(selected_centers)):
                    if has_s1 and i == 0:
                        label_name = "S1"  # 最右邊為 S1
                    else:
                        label_name = f"L{5 - i + (1 if has_s1 else 0)}"  # S1 存在時需修正索引

                    cv2.putText(contour_img, label_name, (cx - 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    centers_coord[label_name] = {"x": cx, "y": cy}

        # 儲存影像
        cv2.imwrite(output_image_path, contour_img)

        return centers_coord

    def process_image(self, image_path, output_path):
        """
        處理 Sacrum 影像並標記 L5，根據斜率變化計算 S1。
        若斜率變化未達閾值，則將最左邊的節點視為 S1。
        """
        img = cv2.imread(image_path)
        processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(processed_image, 50, 255)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40:  # 過濾掉太小的區域
                continue

            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))

        # 按照 x 軸座標排序 (由左到右)
        centers = sorted(centers, key=lambda x: x[0])

        # 計算標點數量
        num_points = len(centers)

        # **在影像上標記中心點**
        for (cx, cy) in centers:
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)  # 綠色點

        # 連接中心點
        for i in range(len(centers) - 1):
            cv2.line(img, centers[i], centers[i + 1], (0, 255, 255), 2)  # 黃色線

        # 計算兩點之間斜率
        slopes = []
        for i in range(len(centers) - 1):
            x1, y1 = centers[i]
            x2, y2 = centers[i + 1]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")
            slopes.append(slope)

        # 找出斜率變化最大的地方
        max_change = 0
        index_of_max_change = 0
        previous_slope = slopes[0]

        for i in range(1, len(slopes)):
            change = abs(slopes[i] - previous_slope)
            if change > max_change:
                max_change = change
                index_of_max_change = i
            previous_slope = slopes[i]

        # 設定斜率變化閾值
        slope_threshold = 0.3  # 斜率變化閾值
        has_s1 = max_change > slope_threshold

        # 決定 S1 位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        if has_s1:
            # 斜率變化超過 threshold，標記最大變化點為 S1
            s1_point = centers[index_of_max_change]
            print(f"檢測到 S1，斜率變化: {max_change}")
        else:
            # 斜率變化不足，改標記最左邊的節點為 S1
            s1_point = centers[0]
            print("斜率變化不足，將最左邊節點標記為 S1")

        # 在影像上標記 S1
        cv2.circle(img, s1_point, 10, (0, 0, 255), -1)  # 紅色標記 S1
        cv2.putText(img, 'S1', (s1_point[0] - 20, s1_point[1] + 35), font, 1, (0, 0, 255), font_thickness)

        # 找出 L5 (S1 左側的點)
        l5_point = None
        s1_index = centers.index(s1_point)
        if s1_index > 0:
            l5_point = centers[s1_index - 1]
            cv2.circle(img, l5_point, 10, (0, 255, 255), -1)  # L5 (黃色標記)
            cv2.putText(img, 'L5', (l5_point[0] - 20, l5_point[1] + 35), font, 1, (0, 255, 255), font_thickness)

        # 若標點數小於 5，顯示警告
        if num_points < 5:
            print(f"Warning: 標記點數量為 {num_points}，顯示缺少 L5")
            warning_text = " L5 NOT FOUND"
            text_size = cv2.getTextSize(warning_text, font, 1, font_thickness)[0]
            
            # 確保文字不會超出範圍
            text_x = max(10, (img.shape[1] - text_size[0]) // 2)
            text_y = max(40, 50)  # 避免太靠邊界

            cv2.putText(img, warning_text, (text_x, text_y), font, 1, (255, 255, 255), font_thickness)

        # 儲存結果影像
        cv2.imwrite(output_path, img)

        # 建立回傳結果：一定有 S1，L5 視情況加入
        result = {
            "S1": {"x": int(s1_point[0]), "y": int(s1_point[1])}
        }

        if l5_point is not None:
            # 正常情況：有找到 L5
            result["L5"] = {"x": int(l5_point[0]), "y": int(l5_point[1])}
        else:
            # 沒有找到 L5，就乾脆不要放 "L5" 這個 key
            # 之後用 "L5" not in data 來判斷
            if hasattr(self, "log"):
                self.log("process_image: 此幀 Sacrum 沒有偵測到 L5，略過 L5 座標寫入")

        return result

    #==============拚接==================
    def find_latest_file(self, prefix, suffix, directory="saved_images"):
        """
        在指定資料夾中找到最大數字的檔案名稱，支援不同副檔名 (jpg, png)。
        """
        if not os.path.exists(directory):
            print(f"目錄 {directory} 不存在，無法搜尋檔案")
            return None  

        # 修正匹配規則，允許不同的影像格式
        pattern = re.compile(rf"{prefix}_(\d+){re.escape(suffix)}$")
        max_index = -1
        latest_file = None

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    latest_file = filename

        return latest_file

    def stitch_images(self):
        """
        讀取最新的 Lumbar & Sacrum 影像及 L5 座標，執行影像拼接，並顯示在 Frame3。
        """
        save_dir = "saved_images"

        # 允許 jpg 和 png 影像
        lumbar_img_file = self.find_latest_file("Lumbar", "_after.jpg") or self.find_latest_file("Lumbar", "_after.png")
        sacrum_img_file = self.find_latest_file("Sacrum", "_after.jpg") or self.find_latest_file("Sacrum", "_after.png")

        # 尋找最新 JSON 檔案
        lumbar_json_file = self.find_latest_file("Lumbar", "_contour.json")
        sacrum_json_file = self.find_latest_file("Sacrum", "_contour.json")

        if not lumbar_img_file or not sacrum_img_file or not lumbar_json_file or not sacrum_json_file:
            print("找不到對應的影像或 JSON 檔案")
            return

        lumbar_img_path = os.path.join(save_dir, lumbar_img_file)
        sacrum_img_path = os.path.join(save_dir, sacrum_img_file)
        lumbar_json_path = os.path.join(save_dir, lumbar_json_file)
        sacrum_json_path = os.path.join(save_dir, sacrum_json_file)

        # 確保 JSON 檔案存在
        try:
            with open(lumbar_json_path, "r", encoding="utf-8") as f:
                lumbar_data = json.load(f)
            with open(sacrum_json_path, "r", encoding="utf-8") as f:
                sacrum_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("JSON 解析失敗，請檢查 JSON 格式或確保文件為 UTF-8 編碼！")
            return

        # **檢查是否有錯誤**
        lumbar_error = "L5" not in lumbar_data or lumbar_data.get("error") == "wrong picture"
        sacrum_error = "L5" not in sacrum_data or sacrum_data.get("error") == "wrong picture"

        if lumbar_error or sacrum_error:
            print("影像有錯誤，顯示 WRONG")

            # **創建一個黑色背景**
            height, width = 600, 400  # 自訂畫布大小
            error_img = np.zeros((height, width, 3), dtype=np.uint8)

            # **在影像中央顯示 "WRONG"**
            error_text = "WRONG"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 3
            text_size = cv2.getTextSize(error_text, font, font_scale, font_thickness)[0]

            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            cv2.putText(error_img, error_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

            # **儲存錯誤影像**
            error_image_path = "error_output.jpg"
            cv2.imwrite(error_image_path, error_img)

            # **在 `self.frame3` 顯示錯誤影像**
            self.frame3.setPixmap(QPixmap(error_image_path))
            self.frame3.setScaledContents(True)

            return  # **終止函數，不執行拼接**

        # **確保 JSON 內有 L5 座標**
        try:
            coord_left = (int(lumbar_data["L5"]["x"]), int(lumbar_data["L5"]["y"]))
            coord_right = (int(sacrum_data["L5"]["x"]), int(sacrum_data["L5"]["y"]))
        except (ValueError, KeyError):
            print("JSON 內的 L5 座標格式錯誤，無法拼接")
            return

        # 從檔名提取 `X` 值
        lumbar_x_match = re.search(r"Lumbar_(\d+)_contour", lumbar_json_file)
        sacrum_x_match = re.search(r"Sacrum_(\d+)_contour", sacrum_json_file)

        if not lumbar_x_match or not sacrum_x_match:
            print("無法從檔名解析 X 值")
            return

        x_value = lumbar_x_match.group(1)  # X 來自 Lumbar_X_contour.json

        # 產生拼接影像檔名
        output_filename = f"{x_value}_stitching.png"
        output_path = os.path.join(save_dir, output_filename)

        # 執行拼接
        self.perform_stitching(lumbar_img_path, sacrum_img_path, coord_left, coord_right, output_path)

        # 顯示拼接後的影像
        self.frame3.setPixmap(QPixmap(output_path))
        self.frame3.setScaledContents(True)

    def perform_stitching(self, image_path_left, image_path_right, coord_left, coord_right, output_path):
        """
        執行影像拼接（使用 L5 座標加權融合）
        """
        img_left = cv2.imread(image_path_left, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(image_path_right, cv2.IMREAD_GRAYSCALE)

        if img_left is None or img_right is None:
            self.log("讀取影像失敗，無法拼接")
            return

        img_left = img_left.squeeze()
        img_right = img_right.squeeze()

        # 計算偏移量（右邊圖貼到左邊圖）
        x_offset = coord_left[0] - coord_right[0]
        y_offset = coord_left[1] - coord_right[1]
        self.log(f"x_offset: {x_offset}, y_offset: {y_offset}")

        # 計算拼接後的影像尺寸
        height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
        width  = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))

        stitched_image = np.zeros((height, width), dtype=np.uint8)
        # 先把左邊整張貼上去
        stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

        # 計算水平重疊區域（可能沒有重疊）
        overlap_start_x = max(x_offset, 0)
        overlap_end_x   = min(img_left.shape[1], img_right.shape[1] + x_offset)
        self.log(f"overlap_start_x: {overlap_start_x}, overlap_end_x: {overlap_end_x}")

        # 建立權重矩陣
        weight_matrix_left  = np.zeros((height, width), dtype=np.float32)
        weight_matrix_right = np.zeros((height, width), dtype=np.float32)

        if overlap_end_x <= overlap_start_x:
            # 幾乎沒有重疊 → 不做漸變，左邊用左圖，右邊用右圖
            for x in range(width):
                if x < img_left.shape[1]:
                    weight_matrix_left[:, x] = 1.0
                else:
                    weight_matrix_right[:, x] = 1.0
        else:
            # 有重疊 → 在重疊區域做線性加權
            denom = float(overlap_end_x - overlap_start_x)
            for x in range(width):
                if x < overlap_start_x:          # 左邊界：全左圖
                    w1, w2 = 1.0, 0.0
                elif overlap_start_x <= x < overlap_end_x:   # 重疊區域：線性漸變
                    alpha = (x - overlap_start_x) / denom
                    w2 = alpha
                    w1 = 1.0 - alpha
                else:                              # 右邊界：全右圖
                    w1, w2 = 0.0, 1.0

                weight_matrix_left[:, x]  = w1
                weight_matrix_right[:, x] = w2

        self.log(
            f"Weight Matrix Left: min={np.min(weight_matrix_left)}, max={np.max(weight_matrix_left)}"
        )
        self.log(
            f"Weight Matrix Right: min={np.min(weight_matrix_right)}, max={np.max(weight_matrix_right)}"
        )

        # 實際加權融合
        for y in range(height):
            for x in range(width):
                left_value = 0
                right_value = 0

                if 0 <= y < img_left.shape[0] and 0 <= x < img_left.shape[1]:
                    left_value = img_left[y, x]

                rx = x - x_offset
                ry = y - y_offset
                if 0 <= ry < img_right.shape[0] and 0 <= rx < img_right.shape[1]:
                    right_value = img_right[ry, rx]

                stitched_image[y, x] = (
                    weight_matrix_left[y, x] * left_value +
                    weight_matrix_right[y, x] * right_value
                ).astype(np.uint8)

        self.log(
            f"Image sizes - Left: {img_left.shape}, Right: {img_right.shape}, Stitched: {stitched_image.shape}"
        )

        cv2.imwrite(output_path, stitched_image)
        self.log(f"Stitched result is saved to {output_path}")

    # ---------- 新增：找最新子資料夾 ----------
    def get_latest_subdir(self, root):
        """回傳 root 底下所有子資料夾（含多層）中最後修改時間最近的資料夾。"""
        try:
            latest = None
            latest_mtime = -1

            for dirpath, dirnames, filenames in os.walk(root):  # 遞迴走訪所有資料夾
                for d in dirnames:
                    full_path = os.path.join(dirpath, d)
                    mtime = os.path.getmtime(full_path)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = full_path

            return latest
        except Exception as e:
            self.log(f"取得最新資料夾失敗：{e}")
            return None
        
    def infer_to_txt(self, image_path: str, session_output_dir: str, labels_dir: str):
        """
        用 YOLO 對單張 image_path 做推論，輸出 txt。
        - project 指向本次 session 的 output_dir
        - 產生的 labels 會在 session_output_dir/predict/labels/
        - 這裡把它搬/改名到 labels_dir/<同名>.txt
        回傳：搬移後的 txt 完整路徑；若失敗回傳 None
        """
        try:
            # 跑 YOLO
            self.yolo_model.predict(
                source=image_path,
                save=False,
                save_txt=True,
                project=session_output_dir,   # e.g. data/realtime/track/{session}
                name="predict",
                exist_ok=True
            )

            # 找剛輸出的 txt（predict/labels/）
            pred_labels_dir = os.path.join(session_output_dir, "predict", "labels")
            base = os.path.splitext(os.path.basename(image_path))[0]  # e.g. frame_000123
            src_txt = os.path.join(pred_labels_dir, base + ".txt")

            if not os.path.exists(src_txt):
                # 有些版本會把副檔名大小寫或路徑有細微差異，兜底找一下
                cand = glob(os.path.join(pred_labels_dir, base + ".*"))
                cand = [p for p in cand if p.lower().endswith(".txt")]
                if not cand:
                    return None
                src_txt = cand[0]

            os.makedirs(labels_dir, exist_ok=True)
            dst_txt = os.path.join(labels_dir, os.path.basename(src_txt))

            # 同名覆蓋
            try:
                sh.move(src_txt, dst_txt)
            except Exception:
                sh.copy(src_txt, dst_txt)

            return dst_txt
        except Exception as e:
            self.log(f"YOLO 產生 txt 失敗：{e}")
            return None
        
    def start_track_thread(self):
        # 防呆：若沒有任何錄影資料夾，給明確提示
        latest_dir, session_folder = self._latest_recording_session()
        if latest_dir is None:
            self.log("找不到錄影資料")
            return

        # 防呆：已在追蹤就不再啟動
        if getattr(self, "is_tracking", False):
            self.log("執行中")
            return

        try:
            # 若有殘留執行緒，先收掉
            if getattr(self, "_track_thread", None) and self._track_thread.isRunning():
                self.stop_track_event.set()
                self._track_thread.quit()
                self._track_thread.wait(2000)
            # 這裡放清空暫存的程式碼！(放在 try 裡，正式啟動前)
            if hasattr(self, "frame_queue"): self.frame_queue.clear()
            if hasattr(self, "label_store"): self.label_store.clear()
            if hasattr(self, "frame_store"): self.frame_store.clear()
            if hasattr(self, "result_queue"): self.result_queue.clear()
            self.next_frame_id = 0
            self.log("[INIT] 清空 RAM 暫存完畢，重置 next_frame_id=0")
                # 切狀態
            self.is_tracking = True
            self.stop_track_event.clear()

            # 按鈕狀態：開始→禁用、停止→啟用
            if hasattr(self, "btn_rt_track_start"): self.btn_rt_track_start.setEnabled(False)
            if hasattr(self, "btn_rt_track_stop"):  self.btn_rt_track_stop.setEnabled(True)

            # 模式判斷：RAM 或 檔案
            use_ram = getattr(self, "rt_use_ram", False)
            if use_ram:
                # RAM 模式需要 YOLO 工人線程
                self.start_yolo_worker()

            # 清楚的啟動訊息
            self.log(f"開始（監看：{latest_dir}，session={session_folder}，模式={'RAM' if use_ram else '檔案'}）")

            # 啟動執行緒（QThread 維持你的寫法）
            self._track_thread = QtCore.QThread(self.root)
            target_func = self._track_loop_ram if use_ram else self.track_images
            self._track_thread.run = target_func   # 維持你原本以 run 綁定目標函式的作法
            # 可選：執行結束時回報
            try:
                self._track_thread.finished.connect(lambda: self.log("追蹤執行緒結束"))
            except Exception:
                pass
            self._track_thread.start()

        except Exception as e:
            # 還原 UI 狀態
            self.is_tracking = False
            if hasattr(self, "btn_rt_track_start"): self.btn_rt_track_start.setEnabled(True)
            if hasattr(self, "btn_rt_track_stop"):  self.btn_rt_track_stop.setEnabled(False)
            self.log(f"啟動失敗：{e}")

    def stop_track_thread(self):
        """停止追蹤（含 RAM 模式的 YOLO 工人）"""
        if not getattr(self, "is_tracking", False):
            self.log("未在執行")
            return
        try:
            # 停止主追蹤 loop
            self.stop_track_event.set()

            if getattr(self, "_track_thread", None):
                try:
                    self._track_thread.quit()
                except Exception:
                    pass
                try:
                    self._track_thread.wait(2000)
                except Exception:
                    pass

            # 若是 RAM 模式，順便停 YOLO 工人線程
            if getattr(self, "rt_use_ram", False):
                try:
                    self.stop_yolo_worker()
                except Exception:
                    pass

            self.log("追蹤已停止")

        finally:
            self.is_tracking = False
            if hasattr(self, "btn_rt_track_start"): self.btn_rt_track_start.setEnabled(True)
            if hasattr(self, "btn_rt_track_stop"):  self.btn_rt_track_stop.setEnabled(False)

    def _yolo_to_centers(self, txt_path, img_shape):
        """
        將 YOLO txt 轉為中心點與框座標。
        支援兩種格式：
        1) class_id cx cy w h [conf]
        2) class_name cx cy w h [conf]  (如 L3 / S2)
        只取出後四欄 (cx,cy,w,h) 做座標換算。
        """
        h, w = img_shape[:2]
        centers = []
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                parts = raw.strip().split()
                if not parts or len(parts) < 5:
                    continue
                # 丟掉第一欄（id 或 名稱），從第二欄開始挑 4 個數字  # [MOD]
                nums = []
                for tok in parts[1:]:
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        nums = []
                        break
                    if len(nums) >= 4:
                        break
                if len(nums) < 4:
                    continue
                x, y, bw, bh = nums[:4]
                cx = int(x * w)
                cy = int(y * h)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                bw_px = int(bw * w)
                bh_px = int(bh * h)
                centers.append((cx, cy, (x1, y1, bw_px, bh_px)))
        centers.sort(key=lambda p: p[0])  # 按 X 座標排序
        return centers

    def auto_label_missing_bones(self, annotations, centers, used_indices, max_move_threshold=50):
        # —— 以下內容與你原本相同（未改動） ——
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

    def annotate_bones(self, image_path, txt_path, output_path, prev_annotations_list=None, max_move_threshold=40, shift_threshold=60):
        # —— 你的原始內容，只有把 yolo_to_centers 改成 self._yolo_to_centers（兩處） ——
        img = cv2.imread(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX

        try:
            centers = self._yolo_to_centers(txt_path, img.shape)  # [MOD]
        except Exception as e:
            self.log(f"無法讀取 {txt_path}：{e}") 
            return prev_annotations_list if prev_annotations_list else None

        annotations = {}
        used_indices = set()

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
            # —— 第一幀策略（原樣保留） ——
            if len(centers) < 2:
                self.log(f"節點不足，跳過 {image_path}")
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

            if s1_index > 0:
                s1_index -= 1

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

            # 反向補 L1→L5
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

            # 反向補 S5→S1
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
                        if i in used_indices or cx >= x_ref - 5:
                            continue
                        dist = np.sqrt((cx - x_ref)**2 + (cy - y_ref)**2)
                        if dist < min_dist and dist < 60:
                            min_dist = dist
                            best_match = (i, cx, cy, box)
                    if best_match:
                        i, cx, cy, box = best_match
                        annotations[next_label] = (cx, cy)
                        used_indices.add(i)
                        x_ref, y_ref = cx, cy
                    else:
                        break

        # 補標遺失節點（再檢查一次）
        annotations = self.auto_label_missing_bones(annotations, centers, used_indices)  # [MOD]

        # 畫框避免重疊（原樣）
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

        # 畫出未使用的框（原樣）
        for i, (cx, cy, (x1, y1, bw, bh)) in enumerate(centers):
            if i not in used_indices:
                too_close = any(abs(px - cx) < 10 and abs(py - cy) < 10 for px, py in drawn_positions)
                if too_close:
                    continue
                drawn_positions.append((cx, cy))
                cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), (0, 0, 255), 2)

        cv2.imwrite(output_path, img)
        return annotations
    def start_yolo_worker(self):
        if self._yolo_thread and self._yolo_thread.is_alive():
            return
        self._yolo_stop.clear()
        t = Thread(target=self._yolo_loop, daemon=True)
        t.start()
        self._yolo_thread = t

    def stop_yolo_worker(self):
        self._yolo_stop.set()

    def _yolo_loop(self):
        while not self._yolo_stop.is_set():
            try:
                fid, img = self.frame_queue.popleft()
            except IndexError:
                time.sleep(0.005); continue

            # --- YOLO 直接吃 ndarray (BGR) ---
            try:
                results = self.yolo_model(img, verbose=False)[0]  # Ultralytics 推論
            except Exception as e:
                self.log(f"YOLO 推論失敗: {e}")
                continue

            h, w = img.shape[:2]
            lines = []
            try:
                boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, "boxes") else []
                for b in boxes:
                    x1, y1, x2, y2 = b[:4]
                    cx = (x1 + x2) / 2.0 / w
                    cy = (y1 + y2) / 2.0 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # 這裡類別 id 先寫 0；如有多類別可改
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            except Exception as e:
                self.log(f"解析 YOLO 結果失敗: {e}")

            yolo_txt = "\n".join(lines)
            self.label_store[fid] = yolo_txt  # → RAM 供 track 使用
            
            # ===== 精簡 YOLO log：只在有偵測框，且每 10 幀印一次 =====
            if len(lines) > 0 and (fid % 10 == 0):
                self.log(f"[YOLO] fid={fid} boxes={len(lines)} label_store={len(self.label_store)}")

            # --- 磁碟鏡射（可關閉） ---
            if getattr(self, "mirror_to_disk", False):
                try:
                    # 找到當前最新的錄影 session 與對應輸出結構
                    latest_dir, session_folder = self._latest_recording_session()
                    if latest_dir:
                        # 產生跟你原本一致的命名（frame_000001.jpg）
                        frame_num = fid
                        img_name = f"frame_{frame_num:06d}.jpg"
                        out_img_path = os.path.join(latest_dir, img_name)
                        cv2.imwrite(out_img_path, img)

                        labels_dir = self._p("realtime", "track", session_folder, "labels")
                        os.makedirs(labels_dir, exist_ok=True)
                        txt_path = os.path.join(labels_dir, f"frame_{frame_num:06d}.txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(yolo_txt)
                except Exception as e:
                    self.log(f"鏡射到磁碟失敗: {e}")


    def track_images(self):
        self.log(f"啟動 Track 模式")

        # 取得最新錄影 session
        latest_dir, session_folder = self._latest_recording_session()
        if latest_dir is None:
            self.log(f"找不到錄影資料")
            return
        self.log(f"最新錄影資料夾：{latest_dir}  (session={session_folder})")

        # 準備輸出資料夾
        base_output_dir = self._p("realtime", "track")
        output_dir      = self._p("realtime", "track", session_folder)
        labels_dir      = self._p("realtime", "track", session_folder, "labels")

        # 取得此 session 的所有影像（支援 png/jpg；以檔名中的序號排序）
        image_paths = []
        for pat in ("frame_*.png", "frame_*.jpg", "after_*.png", "after_*.jpg"):
            image_paths += glob(os.path.join(latest_dir, pat))

        if not image_paths:
            self.log(f"沒有可以處理的影像（frame_*.png/jpg 或 after_*.png/jpg）")
            return

        def _key(p):
            bn = os.path.splitext(os.path.basename(p))[0]  # frame_000123
            # 從最後一段數字取 key，取不到就用 mtime
            try:
                n = re.findall(r"(\d+)$", bn)[-1]
                return int(n)
            except Exception:
                return int(os.path.getmtime(p))

        image_paths.sort(key=_key)

        # 追蹤狀態
        recent_annotations = deque(maxlen=10)
        empty_frame_count = 0

        # 逐張處理
        for image_path in image_paths:
            # 確保有對應的 YOLO txt（集中在 labels/）
            txt_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            if not os.path.exists(txt_path):
                txt_path = self.infer_to_txt(image_path, output_dir, labels_dir)

            # 輸出結果影像（畫上標籤）
            frame_num = _key(image_path)
            output_path = os.path.join(output_dir, f"track_{frame_num:06d}.jpg")

            if not txt_path or not os.path.exists(txt_path):
                # 沒有標註就直接複製原圖
                try:
                    sh.copy(image_path, output_path)
                except Exception as e:
                    self.log(f"複製原圖失敗：{e}")
                    continue
                # 顯示到 frame6（即時處理顯示）
                self._log_emitter.show_image.emit("frame6", output_path)
                self.log(f"追蹤顯示：{os.path.basename(output_path)}")
                time.sleep(0.05)
                empty_frame_count += 1
                continue

            # 用前幀輔助追蹤標註
            prev_info = list(recent_annotations)[::-1]

            annotations = None
            try:
                annotations = self.annotate_bones(image_path, txt_path, output_path, prev_info)  # [MOD]
            except Exception as e:
                self.log(f"標註失敗：{e}", level="error")
                # 兜底：沒有就直接複製原圖
                try:
                    sh.copy(image_path, output_path)
                except:
                    pass
                annotations = None

            # ===== 不管成功/失敗，都在這裡統一寫 nodes / json（用 annotations 為準）=====
            try:
                nodes_txt_path = os.path.join(
                    labels_dir,
                    os.path.splitext(os.path.basename(output_path))[0] + "_nodes.txt"
                )

                # 第一次寫才處理（避免重覆重算同一幀）
                if not os.path.exists(nodes_txt_path):
                    os.makedirs(os.path.dirname(nodes_txt_path), exist_ok=True)

                    # 1) 優先用「追蹤結果 annotations」（這份才是畫在 output_path 上的）
                    anns_names = annotations if isinstance(annotations, dict) else None

                    # 2) 若追蹤結果沒有（例如 annotations=None），才退回單幀再算一次
                    if not anns_names:
                        nodes_tmp_dir = os.path.join(output_dir, "_nodes_tmp")
                        os.makedirs(nodes_tmp_dir, exist_ok=True)
                        tmp_nodes_img = os.path.join(nodes_tmp_dir, os.path.basename(output_path))
                        try:
                            anns_names = self.annotate_bones(
                                image_path, txt_path, tmp_nodes_img,
                                prev_annotations_list=None
                            )
                        except TypeError:
                            anns_names = self.annotate_bones(image_path, txt_path, tmp_nodes_img)

                    # 3) 寫 nodes_txt
                    if isinstance(anns_names, dict) and len(anns_names) > 0:
                        write_nodes_txt(nodes_txt_path, anns_names)

                        # 4) 找 L5（同時支援 'L5' / 5 / '5'）
                        l5_key = None
                        if "L5" in anns_names:
                            l5_key = "L5"
                        elif 5 in anns_names:
                            l5_key = 5
                        elif "5" in anns_names:
                            l5_key = "5"

                        # 5) 有 L5 才輸出 json
                        if l5_key is not None:
                            json_path = os.path.splitext(output_path)[0] + "_contour.json"
                            write_nodes_json(json_path, anns_names)

            except Exception as _e:
                self.log(f"[TRACK] 無法寫入 nodes_txt/JSON: {_e}", level="error")

            if annotations:
                empty_frame_count = 0
                recent_annotations.append(annotations)
            else:
                empty_frame_count += 1
                if empty_frame_count >= 10:
                    self.log(f"連續十幀無偵測框，跳過後續影像")
                    empty_frame_count = 0

            self._log_emitter.show_image.emit("frame6", output_path)
            time.sleep(0.05)
            
        self.log(f"Track 模式完成 → {output_dir}") 

        # ------------------------------------------
        # 即時監控最新子資料夾並逐幀處理（可留可刪）
        # ------------------------------------------
        parent_input_dir = self._p("recording")
        last_seen_folder = None
        processed_frames = set()

        while not self.stop_track_event.is_set():
            latest_dir = self.get_latest_subdir(parent_input_dir)
            if latest_dir is None:
                time.sleep(0.5)
                QtWidgets.QApplication.processEvents()
                continue

            # 偵測到新的最新資料夾 → 重置狀態
            if latest_dir != last_seen_folder:
                self.log(f"切換到最新資料夾：{os.path.basename(latest_dir)}")
                last_seen_folder = latest_dir
                processed_frames.clear()
                recent_annotations.clear()
                empty_frame_count = 0

            # 收集目前有的影像
            image_paths_all = []
            for pat in ("frame_*.png", "frame_*.jpg", "after_*.png", "after_*.jpg"):
                image_paths_all += glob(os.path.join(latest_dir, pat))
            image_paths_all.sort(key=_key)

            # 僅處理「尚未處理」的新幀
            new_frames = [p for p in image_paths_all if p not in processed_frames]
            if not new_frames:
                time.sleep(0.2)
                QtWidgets.QApplication.processEvents()
                continue

            for image_path in new_frames:
                base = os.path.splitext(os.path.basename(image_path))[0]
                txt_path = os.path.join(labels_dir, base + ".txt")

                # 本幀輸出
                frame_num = _key(image_path)
                output_name = f"track_{frame_num:06d}.jpg"
                output_path = os.path.join(output_dir, output_name)

                # 沒有 txt → 先用 YOLO 產生到 labels/
                if not os.path.exists(txt_path):
                    got = self.infer_to_txt(image_path, output_dir, labels_dir)
                    if got:
                        txt_path = got

                # 若仍沒有 txt，就先顯示原圖
                if not os.path.exists(txt_path):
                    try:
                        sh.copy(image_path, output_path)
                    except Exception as e:
                        self.log(f"複製原圖失敗：{e}")
                    self._log_emitter.show_image.emit("frame6", output_path)
                    self.log(f"追蹤顯示：{os.path.basename(output_path)}")
                    time.sleep(0.05)
                    processed_frames.add(image_path)
                    continue

                # 有 txt → 照舊做標註
                prev_info = list(recent_annotations)[::-1]
                try:
                    annotations = self.annotate_bones(image_path, txt_path, output_path, prev_info)  # [MOD]
                except Exception as e:
                    self.log(f"標註失敗：{e}")
                    try:
                        sh.copy(image_path, output_path)
                    except:
                        pass
                    annotations = None

                if annotations:
                    empty_frame_count = 0
                    recent_annotations.append(annotations)
                else:
                    empty_frame_count += 1
                    if empty_frame_count >= 10:
                        self.log(f"連續十幀無偵測框，暫停當前資料夾，等待新資料夾或新幀")
                        empty_frame_count = 0

                # 顯示到 frame6（即時處理顯示）
                self._log_emitter.show_image.emit("frame6", output_path)
                time.sleep(0.05)

                processed_frames.add(image_path)

    def stop_count(self):
        self.stop_count_event.set()
        self.is_counting = False

    def _track_loop_ram(self):
        """
        RAM 版追蹤主迴圈（與 track_images() 呼叫規格一致）：
        - 優先使用 result_queue (fid, img, yolo_txt)；無則用 label_store + frame_store（只吃交集）
        - 寫出 tmp_img/tmp_txt 之後，用 annotate_bones(image_path, txt_path, output_path, prev_info)
        - prev_info = 最近數幀的 annotations（與 track_images 相同語意）
        """
        import os, time, traceback
        import cv2
        from collections import deque
        from PyQt6 import QtWidgets

        self.log("啟動 Track (RAM) 模式")

        # 準備輸出資料夾
        try:
            _, session_folder = self._latest_recording_session()
            output_dir = self._p("realtime", "track", session_folder)
            os.makedirs(output_dir, exist_ok=True)
            self.log(f"[RAM] output_dir={output_dir}")
        except Exception as e:
            self.log(f"取最新錄影資料夾失敗：{e}\n{traceback.format_exc()}", level="error")
            return

        # 和 track_images 一樣：保留近幀的標註結果，提供 prev_info
        recent_annotations = deque(maxlen=10)
        empty_frame_count = 0

        while not self.stop_track_event.is_set():
            try:
                fid = None
                img = None
                yolo_txt = ""
                used_result_queue = False

                # ① 優先吃 result_queue（絕對對齊）
                if hasattr(self, "result_queue"):
                    try:
                        fid, img, yolo_txt = self.result_queue.popleft()
                        used_result_queue = True
                    except Exception:
                        pass

                # ② 後備：label_store + frame_store（只處理交集 fid）
                if not used_result_queue:
                    labels = getattr(self, "label_store", {})
                    frames = getattr(self, "frame_store", {})

                    if not labels:
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.01)
                        continue

                    label_keys = set(labels.keys())
                    frame_keys = set(frames.keys()) if isinstance(frames, dict) else set()
                    common = label_keys & frame_keys

                    if common:
                        fid = min(common)  # 也可改 max(common) 降延遲
                        yolo_txt = labels.pop(fid, "")
                        img = frames.pop(fid, None)
                    else:
                        # 清孤兒 label，避免卡在很舊的 fid
                        if label_keys:
                            oldest = min(label_keys)
                            latest_frame_id = max(frame_keys) if frame_keys else None
                            gc_window = 64
                            should_drop = (latest_frame_id is not None and (latest_frame_id - oldest) > gc_window) \
                                        or (len(labels) > 512)
                            if should_drop:
                                labels.pop(oldest, None)
                                self.log(f"[GC] 丟棄孤兒標註 fid={oldest} (latest={latest_frame_id})")
                        QtWidgets.QApplication.processEvents()
                        time.sleep(0.01)
                        continue

                if img is None:
                    self.log(f"[WARN] 找不到影像 fid={fid}")
                    continue
                img = img.copy()

                # 檔案路徑
                tmp_txt = os.path.join(output_dir, f"tmp_{fid:06d}.txt")
                tmp_img = os.path.join(output_dir, f"tmp_{fid:06d}.jpg")
                out_path = os.path.join(output_dir, f"track_{fid:06d}.jpg")

                # 寫暫存 txt / 圖（annotate_bones 走路徑介面）
                try:
                    with open(tmp_txt, "w", encoding="utf-8") as f:
                        f.write(yolo_txt or "")
                    cv2.imwrite(tmp_img, img)
                except Exception as e:
                    self.log(f"[寫入錯誤] fid={fid}：{e}\n{traceback.format_exc()}", level="error")
                    continue

                # 無偵測 → 直接顯示原圖，避免空畫面
                if not (yolo_txt and yolo_txt.strip()):
                    cv2.imwrite(out_path, img)
                    if hasattr(self, "_log_emitter"):
                        self._log_emitter.show_image.emit("frame6", out_path)
                    #self.log(f"[RAM] 無偵測 → 顯示原圖 fid={fid}")
                    empty_frame_count += 1
                    QtWidgets.QApplication.processEvents()
                    time.sleep(0.02)
                    continue

                # === 關鍵：呼叫方式與 track_images 一致 ===
                prev_info = list(recent_annotations)[::-1]
                annotations = None
                try:
                    # 你的 track_images 用的是 annotate_bones(image_path, txt_path, output_path, prev_info)
                    annotations = self.annotate_bones(tmp_img, tmp_txt, out_path, prev_info)
                except TypeError:
                    # 若舊版僅支援三參數，則降級呼叫
                    annotations = self.annotate_bones(tmp_img, tmp_txt, out_path)
                except Exception as e:
                    self.log(f"annotate_bones() 失敗 fid={fid}: {e}\n{traceback.format_exc()}", level="error")
                    # 保底：輸出原圖
                    try: cv2.imwrite(out_path, img)
                    except: pass
                    annotations = None
                # ===== 寫出 _nodes.txt 與 L5 JSON（改用「真正畫在圖上的 annotations」）=====
                try:
                    nodes_txt_path = os.path.join(
                        output_dir, "labels",
                        os.path.splitext(os.path.basename(out_path))[0] + "_nodes.txt"
                    )

                    # 第一次寫才處理（避免重覆重算同一幀）
                    if not os.path.exists(nodes_txt_path):
                        os.makedirs(os.path.dirname(nodes_txt_path), exist_ok=True)

                        # 1) 優先用「追蹤結果 annotations」（這份才是畫在 out_path 上的）
                        anns_names = annotations if isinstance(annotations, dict) else None

                        # 2) 若追蹤結果沒有（例如 annotations=None），才退回單幀再算一次
                        if not anns_names:
                            nodes_tmp_dir = os.path.join(output_dir, "_nodes_tmp")
                            os.makedirs(nodes_tmp_dir, exist_ok=True)
                            tmp_nodes_img = os.path.join(nodes_tmp_dir, os.path.basename(out_path))

                            try:
                                anns_names = self.annotate_bones(
                                    tmp_img, tmp_txt, tmp_nodes_img,
                                    prev_annotations_list=None
                                )
                            except TypeError:
                                anns_names = self.annotate_bones(tmp_img, tmp_txt, tmp_nodes_img)

                        # 3) 寫 nodes_txt
                        if isinstance(anns_names, dict) and len(anns_names) > 0:
                            write_nodes_txt(nodes_txt_path, anns_names)

                            # 4) 找 L5（同時支援 'L5' / 5 / '5'）
                            l5_key = None
                            if "L5" in anns_names:
                                l5_key = "L5"
                            elif 5 in anns_names:
                                l5_key = 5
                            elif "5" in anns_names:
                                l5_key = "5"

                            if l5_key is not None:
                                json_path = os.path.splitext(out_path)[0] + "_contour.json"
                                # 不再 log 正常成功的 JSON 輸出
                                if not write_nodes_json(json_path, anns_names):
                                    # 只在失敗時印出錯誤
                                    self.log(f"[TRACK-RAM] JSON 寫入失敗: {json_path}", level="error")
                            # 刪除 原本「此幀無 L5」的噪音 log（這是正常狀況，不該出現在 log）
                        else:
                            # 只有真正錯誤才印
                            self.log("[TRACK-RAM] 本幀標註結果為空，略過寫入", level="error")

                except Exception as _e:
                    self.log(f"[TRACK-RAM] 無法寫入 nodes_txt/JSON: {_e}")


                # 與 track_images 相同的後續處理
                if annotations:
                    empty_frame_count = 0
                    recent_annotations.append(annotations)
                else:
                    empty_frame_count += 1
                    if empty_frame_count >= 10:
                        self.log("連續十幀無偵測框，暫緩處理")
                        empty_frame_count = 0

                # 顯示到 frame6
                if os.path.exists(out_path):
                    if hasattr(self, "_log_emitter"):
                        self._log_emitter.show_image.emit("frame6", out_path)
                    src = "result_queue" if used_result_queue else "store"
                    self.log(f"[EMIT] frame6 fid={fid} (src={src}) out={out_path}")
                else:
                    self.log(f"[警告] 未產生輸出圖片 fid={fid}")

                QtWidgets.QApplication.processEvents()
                time.sleep(0.02)

            except Exception as e:
                self.log(f"RAM 追蹤標註失敗：{e}\n{traceback.format_exc()}", level="error")
                break

        self.log("RAM 追蹤執行緒結束")

    def on_rt_judge_best(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.root,
            "選擇腰椎圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path and os.path.isfile(file_path):

            self.rt_lumbar_selected_path = file_path  # ← 儲存路徑！

            # -------------新增：同步紀錄對應 JSON 路徑-------------
            lumbar_json_path = os.path.splitext(file_path)[0] + "_contour.json"
            if os.path.exists(lumbar_json_path):
                self.rt_lumbar_selected_json_path = lumbar_json_path
                self.log(f"已找到腰椎 JSON：{lumbar_json_path}")
            else:
                self.rt_lumbar_selected_json_path = None
                self.log(f"找不到腰椎 JSON：{lumbar_json_path}，之後會改用重偵測")
            # ----------------------------------------------------

            pix = QtGui.QPixmap(file_path)
            self.rt_lumbar_preview.setPixmap(
                pix.scaled(
                    self.rt_lumbar_preview.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.rt_lumbar_preview.show()

            print(f"腰椎圖片已選擇：{file_path}")

    def on_rt_judge_best1(self):
        """
        選擇「最佳薦椎」圖片：
        1. 開檔對話框讓你選擇一張薦椎的 track 圖
        2. 記錄 self.rt_sacrum_selected_path 給之後拼接用
        3. 嘗試找同名的 _contour.json → 設定 self.rt_sacrum_selected_json_path
        4. 在右下角 self.rt_sacrum_preview 顯示縮圖
        """
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.root,
            "選擇薦椎圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path and os.path.isfile(file_path):

            # 1) 記錄薦椎影像路徑
            self.rt_sacrum_selected_path = file_path

            # 2) 嘗試找同名 JSON（例如 track_000004_contour.json）
            sacrum_json_path = os.path.splitext(file_path)[0] + "_contour.json"
            if os.path.exists(sacrum_json_path):
                self.rt_sacrum_selected_json_path = sacrum_json_path
                self.log(f"已找到薦椎 JSON：{sacrum_json_path}")
            else:
                self.rt_sacrum_selected_json_path = None
                self.log(f"找不到薦椎 JSON：{sacrum_json_path}，之後會改用重偵測")

            # 3) 顯示預覽圖到 rt_sacrum_preview
            pix = QtGui.QPixmap(file_path)
            self.rt_sacrum_preview.setPixmap(
                pix.scaled(
                    self.rt_sacrum_preview.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.rt_sacrum_preview.show()

            print(f"薦椎圖片已選擇：{file_path}")

    def on_rt_stitch(self):
        """
        即時模式拼接：
        1. 優先從「同張圖的 JSON」讀 L5 座標
        2. 若 JSON 沒有 L5，再用 find_and_draw_contours / process_image 重偵測
        3. 若兩邊都有 L5，呼叫 perform_stitching 以 L5 對齊並顯示在 frame7
        4. 任一邊失敗就顯示 WRONG，不讓程式當掉
        """

        # --- 檢查是否選過圖 ---
        if not hasattr(self, "rt_lumbar_selected_path"):
            self.log("請先選擇腰椎圖片")
            return
        if not hasattr(self, "rt_sacrum_selected_path"):
            self.log("請先選擇薦椎圖片")
            return

        lumbar_path = self.rt_lumbar_selected_path
        sacrum_path = self.rt_sacrum_selected_path

        # --- Lumbar 顏色設定（跟 run_segmentation_pipeline 一樣，給重偵測用）---
        green_lower, green_upper = np.array([40, 40, 40]), np.array([80, 255, 255])
        red_lower, red_upper = np.array([0, 50, 50]), np.array([10, 255, 255])
        lumbar_color_ranges = [(green_lower, green_upper), (red_lower, red_upper)]
        lumbar_labels_colors = [("conus", (0, 255, 0)), ("bone", (0, 0, 255))]

        # === 1) 先嘗試從「同張圖的 JSON」讀 L5（最穩） ===
        coord_left = None   # 腰椎 L5
        coord_right = None  # 薦椎 L5

        def _get_L5_from_json(json_path):
            """從 contour.json 嘗試取出 L5 座標，容許多種 key 格式。"""
            if not json_path or not os.path.exists(json_path):
                return None
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    d = json.load(f)

                # 支援 'L5' / 'l5' / '5' / 5 等 key
                possible_keys = ["L5", "l5", "5", 5]
                l5 = None
                for k in possible_keys:
                    # 先比對字串 key
                    if isinstance(k, str) and k in d:
                        l5 = d[k]
                        break
                    # 也容忍數字 key
                    if not isinstance(k, str) and k in d:
                        l5 = d[k]
                        break

                if isinstance(l5, dict) and "x" in l5 and "y" in l5:
                    return (int(l5["x"]), int(l5["y"]))
                return None
            except Exception as e:
                self.log(f"JSON 讀取失敗：{json_path}，{e}")
                return None

        # 讀腰椎 L5
        coord_left = _get_L5_from_json(getattr(self, "rt_lumbar_selected_json_path", None))
        # 讀薦椎 L5
        coord_right = _get_L5_from_json(getattr(self, "rt_sacrum_selected_json_path", None))

        # === 2) 若 JSON 沒有 L5，再 fallback 重偵測 ===
        if coord_left is None:
            self.log("腰椎 JSON 沒有 L5，改用重偵測")
            lumbar_tmp = "rt_lumbar_tmp.png"
            lumbar_data = self.find_and_draw_contours(
                lumbar_path, lumbar_tmp, lumbar_color_ranges, lumbar_labels_colors
            )

            def _get_L5_from_dict(d):
                if not isinstance(d, dict) or "error" in d:
                    return None
                l5 = d.get("L5")
                if isinstance(l5, dict) and "x" in l5 and "y" in l5:
                    return (int(l5["x"]), int(l5["y"]))
                return None

            coord_left = _get_L5_from_dict(lumbar_data)

        if coord_right is None:
            self.log("薦椎 JSON 沒有 L5，改用重偵測")
            sacrum_tmp = "rt_sacrum_tmp.png"
            sacrum_data = self.process_image(sacrum_path, sacrum_tmp)

            def _get_L5_from_sacrum(d):
                if not isinstance(d, dict):
                    return None
                l5 = d.get("L5")
                if isinstance(l5, dict) and "x" in l5 and "y" in l5:
                    return (int(l5["x"]), int(l5["y"]))
                return None

            coord_right = _get_L5_from_sacrum(sacrum_data)

        # === 3) 如果還是沒有 L5，就顯示 WRONG，不做拼接 ===
        if coord_left is None or coord_right is None:
            self.log("任一側 L5 未偵測到，無法拼接")
            self._show_wrong(self.frame7)
            return

        # === 4) 以 L5 座標執行拼接，產生 out_path ===
        rt_stitch_dir = os.path.join("data", "realtime", "stitch")
        os.makedirs(rt_stitch_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(lumbar_path))[0]
        out_path = os.path.join(rt_stitch_dir, f"{base_name}_rt_stitched.png")

        # 實際執行拼接
        self.perform_stitching(lumbar_path, sacrum_path, coord_left, coord_right, out_path)

        # 防呆：確認輸出檔真的存在
        if not os.path.exists(out_path):
            self.log("拼接失敗：找不到輸出影像", level="error")
            self._show_wrong(self.frame7)
            return

        # === 5) 顯示到 frame7 ===
        pix = QtGui.QPixmap(out_path)
        self.frame7.setPixmap(
            pix.scaled(
                self.frame7.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )
        )
        self.frame7.setText("")
        self.log(f"✔ 即時拼接完成：{out_path}")


    def on_rt_stitch(self):
        """
        即時模式拼接：
        1. 用 JSON 或 fallback 偵測兩邊 L5
        2. 成功後呼叫 perform_stitching()
        3. 顯示拼接結果到 frame7
        """

        # --- 檢查是否有選圖 ---
        if not hasattr(self, "rt_lumbar_selected_path"):
            self.log("請先選擇腰椎圖片")
            return
        if not hasattr(self, "rt_sacrum_selected_path"):
            self.log("請先選擇薦椎圖片")
            return

        lumbar_path = self.rt_lumbar_selected_path
        sacrum_path = self.rt_sacrum_selected_path

        # === 1) 嘗試讀 JSON 取 L5 ===
        def _get_L5_from_json(json_path):
            if not json_path or not os.path.exists(json_path):
                return None
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                l5 = d.get("L5")
                if isinstance(l5, dict) and "x" in l5 and "y" in l5:
                    return (int(l5["x"]), int(l5["y"]))
                return None
            except Exception as e:
                self.log(f"JSON 讀取失敗：{json_path}，{e}")
                return None

        coord_left = _get_L5_from_json(getattr(self, "rt_lumbar_selected_json_path", None))
        coord_right = _get_L5_from_json(getattr(self, "rt_sacrum_selected_json_path", None))

        # === 2) 若 JSON 沒有 L5 → fallback 重偵測（你原本的流程） ===
        if coord_left is None:
            self.log("腰椎 JSON 沒有 L5，改用重偵測")
            lumbar_tmp = "rt_lumbar_tmp.png"
            lumbar_data = self.find_and_draw_contours(
                lumbar_path, lumbar_tmp,
                [(np.array([40,40,40]), np.array([80,255,255])),
                (np.array([0,50,50]),  np.array([10,255,255]))],
                [("conus",(0,255,0)),("bone",(0,0,255))]
            )
            l5 = lumbar_data.get("L5") if isinstance(lumbar_data, dict) else None
            coord_left = (int(l5["x"]), int(l5["y"])) if l5 else None

        if coord_right is None:
            self.log("薦椎 JSON 沒有 L5，改用重偵測")
            sacrum_tmp = "rt_sacrum_tmp.png"
            sacrum_data = self.process_image(sacrum_path, sacrum_tmp)
            l5 = sacrum_data.get("L5") if isinstance(sacrum_data, dict) else None
            coord_right = (int(l5["x"]), int(l5["y"])) if l5 else None

        # === 3) 若任一側沒有 L5 → 顯示 WRONG ===
        if coord_left is None or coord_right is None:
            self.log("❌ 任一側 L5 未偵測到，無法拼接")
            self._show_wrong(self.frame7)
            return

        # === 4) 產生輸出檔名 ===
        # 取 lumbar 圖片的編號
        basename = os.path.basename(lumbar_path)
        match = re.search(r"(\d+)", basename)
        num = match.group(1) if match else "000"

        save_dir = "saved_images"
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, f"{num}_stitching_rt.png")

        # === 5) 執行拼接 ===
        self.perform_stitching(lumbar_path, sacrum_path, coord_left, coord_right, out_path)

        # === 6) 顯示到 frame7 ===
        pix = QtGui.QPixmap(out_path)
        self.frame7.setPixmap(
            pix.scaled(
                self.frame7.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )
        )
        self.frame7.setText("")
        self.log(f"✔ 即時拼接完成：{out_path}")


    def _show_wrong(self, frame):
        """
        在指定的 QLabel(frame) 顯示一張寫著 WRONG 的錯誤圖
        """
        height, width = 400, 600
        img = np.zeros((height, width, 3), dtype=np.uint8)

        text = "WRONG"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2

        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 255), thickness)

        tmp_path = "rt_wrong_tmp.png"
        cv2.imwrite(tmp_path, img)

        frame.setPixmap(QtGui.QPixmap(tmp_path))
        frame.setScaledContents(True)

    def count_images(self):
        """RAM 版計數流程（自動循環版）"""

        if self.is_counting:
            self.log("計數已在執行中")
            return

        if not hasattr(self, "frame_queue") or len(self.frame_queue) == 0:
            self.log("RAM 暫無影像")
            return

        self.is_counting = True
        self.stop_count_event.clear()

        def loop_count():
            while not self.stop_count_event.is_set():
                try:
                    if not hasattr(self, "frame_queue") or len(self.frame_queue) == 0:
                        self.log("RAM 暫無影像，等待中...")
                        time.sleep(0.5)
                        continue

                    fid, img_bgr = self.frame_queue[-1]
                    img_bgr = img_bgr.copy()

                    # 1️暫存輸入圖
                    count_dir = self._p("realtime", "count")
                    os.makedirs(count_dir, exist_ok=True)
                    input_path = os.path.join(count_dir, f"count_input_{fid:06d}.jpg")
                    cv2.imwrite(input_path, img_bgr)

                    # 2️YOLO txt 輸出
                    yolo_dir = self._p("realtime", "count", "labels")
                    os.makedirs(yolo_dir, exist_ok=True)
                    txt_path = os.path.join(yolo_dir, f"count_{fid:06d}.txt")
                    results = self.yolo_model(img_bgr, verbose=False)
                    results[0].save_txt(txt_path)

                    # 3️標註骨節點
                    labels_dict, centers, used_indices = annotate_bones_count(input_path, txt_path, None)

                    if labels_dict is None:
                        self.log("骨骼偵測失敗，無法標註")
                        time.sleep(0.5)
                        continue
                    
                    # 產出椎體清單（nodes_txt），供後續『判斷最佳』使用
                    nodes_txt_path = os.path.join(yolo_dir, f"count_{fid:06d}_nodes.txt")
                    try:
                        write_nodes_txt(nodes_txt_path, labels_dict)
                    except Exception as _e:
                        self.log(f"[COUNT] 無法寫入")

                    # 繪製輸出圖
                    output_path = os.path.join(count_dir, f"count_{fid:06d}.jpg")
                    draw_annotations(input_path, labels_dict, centers, used_indices, output_path)

                    # 顯示影像
                    self._log_emitter.show_image.emit("frame6", output_path)

                    # 更新
                    self.next_frame_id = fid + 1
                    self.log(f"[COUNT] {output_path} → 標註節點: {len(labels_dict)}")

                    # 每輪間隔（可調整速度）
                    #time.sleep(0.5)

                except Exception as e:
                    import traceback
                    self.log(f"計數模式發生例外: {e}")
                    self.log(traceback.format_exc())
                    time.sleep(1)

            self.is_counting = False
            self.log("計數循環已停止")
                                                
        threading.Thread(target=loop_count, daemon=True).start()
