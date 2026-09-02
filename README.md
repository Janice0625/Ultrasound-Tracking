<<<<<<< HEAD
# Pediatric Spine Ultrasound Real-time Tracking and Stitching System
### 小兒脊椎超音波影像即時追蹤與拼接系統

---

## 📌 專案簡介 (Project Overview)
[cite_start]新生兒的脊髓疾病如**脊髓牽引症 (Tethered Cord Syndrome, TCS)** 若未能於出生初期即時篩檢，可能導致永久性神經損傷 [cite: 124, 293][cite_start]。然而，現行臨床的脊椎超音波檢查受限於探頭掃描視野狹小，必須依靠醫師經驗以肉眼分段判讀，造成高度的認知與操作負荷 [cite: 204, 206, 293]。

[cite_start]本研究以影像處理與深度學習為核心，開發了一套**端到端 (End-to-End) 的小兒脊椎超音波影像即時追蹤與輔助診斷系統** [cite: 294][cite_start]。本系統結合了 **U-Net 分割 [cite: 338, 353][cite_start]、YOLOv10 即時物件偵測 [cite: 251, 353] [cite_start]以及動態影像拼接演算法 [cite: 340][cite_start]**，能在醫師操作超音波探頭時，即時標註椎體輪廓、進行椎骨自動計數與追蹤 [cite: 284, 285, 354][cite_start]，並將腰椎與薦椎無縫拼接成完整的脊柱圖像 [cite: 218, 340][cite_start]，大幅降低臨床判讀門檻 [cite: 294, 352]。

---

## 🚀 核心技術亮點 (Key Features & Tech Stack)

1. **U-Net 區域分割模組 (Spine Segmentation)**
   * [cite_start]負責精準分割超音波影像中的腰椎 (Lumbar) 與薦椎 (Sacrum) 區域 [cite: 296, 297, 338][cite_start]，建立高品質的脊柱遮罩，為後續的影像拼接提供穩固的解剖結構基礎 [cite: 338]。
2. **YOLOv10 即時椎體偵測 (Real-time Bone Detection)**
   * [cite_start]引入最新的 **YOLOv10** 模型，在動態超音波視訊串流中快速定位個別椎體位置 [cite: 251, 339][cite_start]，並支援雙重臨床操作模式 [cite: 354]：
     * [cite_start]**計數模式 (Count Mode)**：依據臨床習慣，將畫面最右側（薦椎端）標註為 1 號，並向左依序編號自動計數 [cite: 263, 264, 265]。
     * [cite_start]**追蹤模式 (Track Mode)**：在探頭移動過程中動態鎖定並追蹤椎體 [cite: 254, 354][cite_start]，利用斜率變化自動偵測與定位 L5 與 S1 關鍵節點 [cite: 259, 261, 280]。
3. **骨頭輪廓影像拼接演算法 (Image Stitching Pipeline)**
   * [cite_start]利用 **Canny 邊緣偵測** 擷取椎骨邊緣特徵 [cite: 340][cite_start]，並在腰椎與薦椎重疊區域採用**加權融合 (Weighted Blending) 演算法** [cite: 340][cite_start]，實現影像無縫拼接，生成全景平面脊柱圖像 [cite: 218, 340]。
4. **自動錯誤檢測機制 (Auto Error Screening)**
   * [cite_start]針對臨床採集痛點，開發出自動錯誤排除演算法 [cite: 277, 341][cite_start]。能自動辨識模糊、探頭滑移或椎骨節數不足等不符診斷標準的影像，即時提示 `Wrong Picture` [cite: 278, 282][cite_start]，協助篩選出最具參考價值的標準影像 [cite: 283]。

---

## 🛠️ 系統架構與操作流程 (System Workflow)

[cite_start]本系統提供兩大靈活的操作模式，以適應不同的臨床情境 [cite: 235, 354]：

```text
[ 超音波影像輸入 / Ultrasound Stream ]
                  │
                  ▼
          ┌───────────────┐
          │ 選擇運行模式   │
          └───────┬───────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
【 拼接模式 (Stitch) 】   【 即時模式 (Real-time) 】
  (處理已有影像 [cite: 238])       (動態錄影串流 [cite: 249, 250])
        │                   │
  影像前處理 [cite: 239]             過 YOLOv10 模型 
        │                   ├───────────────┐
  過 U-Net 模型 [cite: 240, 328]     ▼               ▼
        │             計數模式 (Count)  追蹤模式 (Track)
  椎體輪廓標註 [cite: 241, 329]    [cite: 263, 331]      [cite: 254, 332]
        │                   │               │
  邊緣輪廓提取與拼接 [cite: 340]  最右編號為1    利用斜率定位椎骨
        │             向左依序編號    即時標註與追蹤
        ▼             [cite: 264, 265]     [cite: 259, 280, 285]
┌───────────────────────────────────────────┐
│ PyQt6 醫療級展示介面 (即時繪製與標註連線) [cite: 1, 284, 306] │
└───────────────────────────────────────────┘
=======
# Ultrasound-Tracking
Real-time pediatric spinal ultrasound tracking using YOLOv10 and image stitching
>>>>>>> b575b142a0821af1d925316ac79303b75e2cf545
