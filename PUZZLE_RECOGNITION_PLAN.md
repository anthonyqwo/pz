# 拼圖形狀辨識系統開發規劃

## 1. 專案目標

本專案目標是開發一套可透過照片辨識黑色拼圖片應放置於底板哪個位置的系統。

使用者會拍攝：

- 空底板照片：用來建立底板位置資料庫。
- 拼圖照片：包含一片或多片全黑拼圖片。

系統會根據拼圖片的外輪廓形狀，與底板上已編號的位置進行比對，最後輸出拼圖片對應的底板位置 ID、旋轉角度與信心分數。

核心概念：

```text
底板位置先編號與建模
-> 拍攝黑色拼圖片
-> 擷取拼圖片形狀
-> 與所有底板 slot 形狀比對
-> 輸出最可能的 slot ID
```

## 2. 使用情境與假設

### 2.1 初版假設

MVP 階段先採用以下限制，以便快速驗證技術可行性：

- 底板為固定款式。
- 相機位置盡量固定。
- 拍攝時底板完整出現在畫面中。
- 拼圖片為全黑或接近黑色。
- 底板與拼圖片有足夠顏色對比。
- 初期先支援單片拼圖辨識。
- 底板位置可事先建立模板。
- 不依賴拼圖片上的文字、花紋、QR code 或顏色資訊。

### 2.2 後續擴充目標

後續版本可逐步支援：

- 多片拼圖同時辨識。
- 不同底板款式。
- 半自動或全自動建立底板 slot。
- 手機拍攝角度較自由的情境。
- 疑似錯誤或低信心結果提示。
- Web UI 顯示辨識結果。
- API 服務化。

## 3. 系統整體流程

系統分為兩條主要流程：

```text
流程 A：建立底板資料

空底板照片
-> 偵測底板定位點
-> 透視校正
-> 建立 slot 輪廓與 mask
-> 編號
-> 儲存 board config


流程 B：辨識拼圖位置

拼圖照片
-> 偵測底板定位點
-> 透視校正
-> 偵測黑色拼圖片
-> 擷取拼圖片輪廓與 mask
-> 與 slot 模板比對
-> 輸出 matched_slot_id
```

## 4. 技術路線

### 4.1 優先採用傳統電腦視覺

初期不建議直接使用深度學習。原因如下：

- 拼圖為全黑，傳統影像分割較容易處理。
- 形狀匹配問題可用輪廓與 mask 比對完成。
- 傳統 CV 需要的訓練資料少。
- Debug 成本較低，可解釋性較高。
- MVP 可更快驗證核心假設。

建議使用：

- Python
- OpenCV
- NumPy
- SciPy
- FastAPI 或 Flask
- 簡單 Web UI 或命令列工具

### 4.2 後期才考慮 AI 模型

若遇到以下情況，再考慮加入 AI：

- 光影變化過大，threshold 不穩。
- 拼圖片不是純黑。
- 背景干擾嚴重。
- 拼圖片邊界模糊。
- 需要支援大量不同底板。
- 有大量標註資料可用。

可能方案：

- YOLO segmentation
- Mask R-CNN
- Segment Anything Model
- Siamese network 形狀相似度模型
- CNN 分類拼圖片對應 slot

## 5. 拍攝與硬體建議

影像品質會直接影響辨識準確率，因此拍攝規格很重要。

### 5.1 拍攝環境

建議：

- 相機固定高度與角度。
- 底板放在固定位置。
- 光源均勻，避免強烈陰影。
- 避免黑色拼圖片表面反光。
- 底板背景與拼圖片顏色有高對比。
- 每張照片都包含完整底板與定位點。

### 5.2 底板定位點

為了做透視校正，建議在底板四角加入定位標記。

可選方案：

| 方案 | 優點 | 缺點 |
| --- | --- | --- |
| ArUco marker | 偵測穩定，可取得 ID 與角點 | 需要印刷 marker |
| 四個高對比圓點 | 簡單直覺 | 需自行判斷角點順序 |
| 棋盤格角點 | 精度高 | 外觀較干擾 |
| 固定顏色角標 | 易於製作 | 光線變化下需調整參數 |

MVP 建議使用 ArUco marker 或四個高對比圓點。

## 6. 底板校正設計

### 6.1 目的

不同照片會有拍攝角度、距離與透視變形。校正的目標是把每張照片轉成固定大小的俯視圖。

例如：

```text
原始照片
-> 找到底板四角
-> perspective transform
-> 轉成 2000 x 2000 的標準底板圖
```

### 6.2 輸入

- 原始照片
- 底板定位點設定
- 目標輸出尺寸

### 6.3 輸出

```json
{
  "board_id": "board_001",
  "rectified_width": 2000,
  "rectified_height": 2000,
  "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "rectified_image_path": "data/boards/board_001/empty_rectified.png"
}
```

### 6.4 實作重點

- 偵測四個定位點。
- 將四角排序為左上、右上、右下、左下。
- 使用 OpenCV `cv2.getPerspectiveTransform`。
- 使用 OpenCV `cv2.warpPerspective`。
- 每張輸入照片都轉換到同一個底板座標系。

## 7. 建立底板 Slot 資料庫

### 7.1 Slot 定義

slot 是底板上每個可放置拼圖片的位置。

每個 slot 需要有唯一 ID，例如：

```text
slot_001
slot_002
slot_003
```

### 7.2 建立方式

#### 方式 A：手動標註

由人工在校正後的空底板圖上框選或描繪每個 slot 輪廓。

優點：

- 最穩定。
- 不容易受光影影響。
- 適合初版開發。

缺點：

- 每種底板都需要人工處理一次。

#### 方式 B：半自動偵測

系統先從空底板照片找出可能的 slot 輪廓，再由人工確認與修正。

優點：

- 比全手動快。
- 可保留人工品質控制。

缺點：

- 需要做簡單標註工具或檢查介面。

#### 方式 C：全自動偵測

系統自動從空底板照片偵測所有 slot，並依位置排序編號。

優點：

- 使用流程最自動化。

缺點：

- 對底板設計、光影與背景要求較高。
- 初版不建議直接做。

### 7.3 MVP 建議

MVP 採用半自動或手動方式。

推薦流程：

```text
上傳空底板照片
-> 透視校正
-> 使用者標註每個 slot
-> 系統產生 mask、contour、area、bbox、center
-> 儲存 board_config.json
```

### 7.4 Slot 資料格式

```json
{
  "slot_id": "slot_017",
  "mask_path": "slots/slot_017_mask.png",
  "contour": [[120, 80], [135, 82], [150, 100]],
  "area": 8420,
  "perimeter": 388.4,
  "bbox": [110, 70, 90, 120],
  "center": [155, 130],
  "rotation_mode": "any",
  "enabled": true
}
```

### 7.5 Board Config 格式

```json
{
  "board_id": "board_001",
  "version": 1,
  "rectified_size": [2000, 2000],
  "marker_type": "aruco",
  "slots": [
    {
      "slot_id": "slot_001",
      "mask_path": "slots/slot_001_mask.png",
      "area": 8420,
      "perimeter": 388.4,
      "bbox": [110, 70, 90, 120],
      "center": [155, 130],
      "rotation_mode": "any",
      "enabled": true
    }
  ]
}
```

## 8. 拼圖片偵測

### 8.1 輸入

- 包含黑色拼圖片的照片。
- 對應底板的 board config。

### 8.2 處理流程

```text
原始照片
-> 透視校正
-> 轉灰階或 HSV
-> threshold 找黑色區域
-> morphology open 去除小雜訊
-> morphology close 補齊邊緣破洞
-> findContours 找拼圖片輪廓
-> 過濾不合理 contour
-> 輸出 piece candidates
```

### 8.3 黑色區域偵測

可使用灰階：

```text
gray < threshold
```

或使用 HSV：

```text
V < threshold
S 可視情況限制
```

初始參數建議：

```text
gray_threshold = 60
min_piece_area = 500
max_piece_area = 200000
```

實際數值需依照片解析度與底板大小調整。

### 8.4 Contour 過濾條件

可用以下條件過濾雜訊：

- 面積不能太小。
- 面積不能大於合理拼圖大小。
- bbox 長寬不能太極端。
- contour 必須在底板有效區域內。
- contour 外接矩形不能超出底板邊界太多。
- contour solidity 不應過低。

### 8.5 Piece 資料格式

```json
{
  "piece_id": "piece_001",
  "mask_path": "outputs/debug/piece_001_mask.png",
  "contour": [[300, 240], [312, 250], [340, 260]],
  "area": 8350,
  "perimeter": 381.2,
  "bbox": [300, 240, 100, 130],
  "center": [350, 305]
}
```

## 9. 形狀匹配策略

形狀匹配分成三層：快速篩選、輪廓相似度、mask IoU 精比對。

### 9.1 第一層：快速篩選

先用簡單特徵縮小候選 slot 數量。

條件：

- 面積接近。
- 周長接近。
- bbox 長寬比接近。
- contour 粗略形狀接近。

範例：

```text
abs(piece_area - slot_area) / slot_area < 0.15
```

若 slot 數量很多，可先只保留前 20 個候選。

### 9.2 第二層：輪廓相似度

使用 OpenCV `cv2.matchShapes`：

```python
score = cv2.matchShapes(piece_contour, slot_contour, cv2.CONTOURS_MATCH_I1, 0)
```

特性：

- 分數越小越相似。
- 可作為候選排序。
- 不建議作為唯一判斷標準。

### 9.3 第三層：Mask IoU 精比對

將拼圖片 mask 旋轉到不同角度，與 slot mask 疊合，計算 IoU。

IoU 定義：

```text
IoU = intersection_area / union_area
```

比對流程：

```text
piece mask
-> 平移到 slot center
-> 旋轉 0, 5, 10, ..., 355 度
-> 每個角度計算 IoU
-> 找到最佳角度與最高 IoU
```

若需要更精準：

```text
第一輪：每 5 度粗掃
第二輪：在最佳角度 ±5 度內每 1 度細掃
```

### 9.4 鏡像處理

若拼圖片可能正反面翻轉，需額外測試鏡像版本。

比對模式：

```text
normal
mirror_x
mirror_y
```

輸出時需標記是否為鏡像匹配。

```json
{
  "matched_slot_id": "slot_017",
  "rotation": 92,
  "mirrored": false,
  "iou": 0.94
}
```

## 10. 信心分數與結果判斷

### 10.1 基本指標

每個候選 slot 產生：

- `iou`
- `shape_score`
- `area_diff_ratio`
- `rotation`
- `rank`

### 10.2 信心分數設計

可將不同指標合併成 confidence：

```text
confidence = 0.7 * iou_score + 0.2 * shape_score_normalized + 0.1 * area_score
```

其中：

- IoU 越高越好。
- shape score 需正規化為越高越好。
- area diff 越小越好。

MVP 可先直接以 IoU 作為主要信心分數。

### 10.3 判定狀態

建議結果分成三種：

| 狀態 | 條件 | 意義 |
| --- | --- | --- |
| confident | 最佳分數高，且明顯贏過第二名 | 可自動採用 |
| ambiguous | 第一名與第二名分數接近 | 需要人工確認 |
| rejected | 最佳分數過低 | 可能偵測錯誤或沒有匹配 slot |

初始門檻：

```text
confident:
  best_iou >= 0.85
  best_iou - second_iou >= 0.08

ambiguous:
  best_iou >= 0.75
  best_iou - second_iou < 0.08

rejected:
  best_iou < 0.75
```

實際門檻需用測試照片調整。

### 10.4 輸出格式

```json
{
  "piece_id": "piece_001",
  "matched_slot_id": "slot_017",
  "rotation": 92,
  "mirrored": false,
  "confidence": 0.91,
  "iou": 0.94,
  "shape_score": 0.021,
  "area_diff_ratio": 0.008,
  "status": "confident",
  "candidates": [
    {
      "slot_id": "slot_017",
      "iou": 0.94,
      "rotation": 92
    },
    {
      "slot_id": "slot_021",
      "iou": 0.81,
      "rotation": 270
    }
  ]
}
```

## 11. 多片拼圖處理

### 11.1 單片到多片的差異

單片辨識只需要找最佳 slot。

多片辨識需要處理：

- 一張圖中有多個黑色 contour。
- 多個 piece 可能匹配到同一個 slot。
- 相鄰拼圖片可能接觸導致 contour 合併。
- 部分拼圖片可能被遮擋或超出底板。

### 11.2 多片處理流程

```text
偵測所有 piece contours
-> 每片 piece 產生候選 slot 排名
-> 建立 piece-slot 分數矩陣
-> 使用 Hungarian algorithm 做全域最佳配對
-> 輸出每片 piece 的唯一 slot
```

### 11.3 分數矩陣範例

```text
             slot_001   slot_002   slot_003
piece_001      0.91       0.42       0.30
piece_002      0.50       0.88       0.52
piece_003      0.33       0.49       0.93
```

每個 piece 最後只能對應一個 slot，每個 slot 也只能被使用一次。

### 11.4 Contour 黏連問題

如果兩片黑色拼圖互相接觸，threshold 後可能會變成一個大 contour。

可能處理方式：

- 要求拍攝時拼圖片不要互相接觸。
- 使用 watershed segmentation。
- 根據 concavity 嘗試切割輪廓。
- 後期使用 instance segmentation 模型。

MVP 階段建議先規定拼圖片彼此不要接觸。

## 12. Debug 與視覺化

影像辨識系統一定要保留 debug 圖，否則很難調參。

### 12.1 建議輸出

每次辨識輸出：

- 原圖。
- 校正後圖片。
- 黑色 threshold mask。
- morphology 後 mask。
- 偵測到的 piece contour。
- 每個 piece 的候選 slot 分數。
- 最終匹配 overlay 圖。

### 12.2 Overlay 圖內容

可視覺化：

- piece contour：紅色。
- matched slot contour：綠色。
- slot ID。
- confidence。
- rotation。
- ambiguous / rejected 狀態。

### 12.3 Debug 目錄

```text
outputs/
  debug/
    run_20260522_001/
      input.jpg
      rectified.png
      black_mask.png
      cleaned_mask.png
      pieces_overlay.png
      match_overlay.png
      result.json
```

## 13. API 規劃

### 13.1 建立底板

```http
POST /boards
```

用途：

建立新的底板資料。

輸入：

```json
{
  "board_id": "board_001",
  "image": "base64_or_file_upload",
  "rectified_size": [2000, 2000],
  "marker_type": "aruco"
}
```

輸出：

```json
{
  "board_id": "board_001",
  "status": "created",
  "rectified_image_path": "data/boards/board_001/empty_rectified.png"
}
```

### 13.2 新增或更新 Slot

```http
POST /boards/{board_id}/slots
```

用途：

新增人工標註的 slot。

輸入：

```json
{
  "slot_id": "slot_017",
  "polygon": [[120, 80], [135, 82], [150, 100]]
}
```

輸出：

```json
{
  "slot_id": "slot_017",
  "status": "saved"
}
```

### 13.3 辨識拼圖

```http
POST /recognize
```

用途：

辨識照片中的拼圖片應對應哪個 slot。

輸入：

```json
{
  "board_id": "board_001",
  "image": "base64_or_file_upload",
  "options": {
    "allow_mirror": false,
    "max_candidates": 20,
    "debug": true
  }
}
```

輸出：

```json
{
  "board_id": "board_001",
  "status": "ok",
  "pieces": [
    {
      "piece_id": "piece_001",
      "matched_slot_id": "slot_017",
      "rotation": 92,
      "mirrored": false,
      "confidence": 0.91,
      "status": "confident"
    }
  ],
  "debug": {
    "rectified_image_path": "outputs/debug/run_001/rectified.png",
    "match_overlay_path": "outputs/debug/run_001/match_overlay.png"
  }
}
```

### 13.4 取得底板設定

```http
GET /boards/{board_id}
```

輸出：

```json
{
  "board_id": "board_001",
  "rectified_size": [2000, 2000],
  "slot_count": 120,
  "created_at": "2026-05-22T00:00:00+08:00"
}
```

## 14. 專案目錄規劃

建議目錄結構：

```text
puzzle-recognition/
  README.md
  requirements.txt
  pyproject.toml

  data/
    boards/
      board_001/
        board_config.json
        empty_original.jpg
        empty_rectified.png
        slots/
          slot_001_mask.png
          slot_002_mask.png
    samples/
      input_001.jpg
      input_002.jpg

  src/
    puzzle_recognition/
      __init__.py
      calibration.py
      board_builder.py
      piece_detector.py
      shape_matcher.py
      recognizer.py
      visualization.py
      config.py

  app/
    api.py
    web_ui/

  scripts/
    build_board.py
    recognize_image.py
    debug_threshold.py

  tests/
    test_calibration.py
    test_piece_detector.py
    test_shape_matcher.py

  outputs/
    debug/
    results/
```

## 15. 模組設計

### 15.1 `calibration.py`

職責：

- 偵測底板定位點。
- 排序四角座標。
- 產生 perspective transform matrix。
- 將照片轉成標準俯視圖。

主要函式：

```python
def detect_board_corners(image) -> list[tuple[float, float]]:
    ...

def rectify_board(image, corners, output_size) -> tuple[np.ndarray, np.ndarray]:
    ...
```

### 15.2 `board_builder.py`

職責：

- 根據空底板建立 board config。
- 由 polygon 產生 slot mask。
- 計算 slot 特徵。
- 儲存 slot 資料。

主要函式：

```python
def create_slot_from_polygon(slot_id, polygon, image_size) -> dict:
    ...

def save_board_config(board_id, slots, output_dir) -> None:
    ...
```

### 15.3 `piece_detector.py`

職責：

- 從校正後照片中偵測黑色拼圖片。
- 產生 piece mask、contour 與特徵。

主要函式：

```python
def detect_black_pieces(rectified_image, config) -> list[dict]:
    ...

def clean_black_mask(mask, kernel_size) -> np.ndarray:
    ...
```

### 15.4 `shape_matcher.py`

職責：

- 快速篩選候選 slot。
- 計算 contour similarity。
- 計算旋轉後 mask IoU。
- 輸出最佳匹配結果。

主要函式：

```python
def find_candidate_slots(piece, slots, max_candidates=20) -> list[dict]:
    ...

def match_piece_to_slot(piece, slot, options) -> dict:
    ...

def match_piece_to_board(piece, board_config, options) -> dict:
    ...
```

### 15.5 `recognizer.py`

職責：

- 串接完整辨識流程。
- 載入 board config。
- 輸出結果 JSON。

主要函式：

```python
def recognize(image_path, board_id, options) -> dict:
    ...
```

### 15.6 `visualization.py`

職責：

- 繪製 debug overlay。
- 輸出 threshold mask。
- 輸出匹配結果圖。

主要函式：

```python
def draw_piece_contours(image, pieces) -> np.ndarray:
    ...

def draw_match_results(image, results, board_config) -> np.ndarray:
    ...
```

## 16. 開發里程碑

### 16.1 MVP 0：實驗腳本

目標：

驗證黑色拼圖片能否穩定從照片分割出來。

任務：

- 寫 threshold 測試腳本。
- 輸出黑色 mask。
- 找 contour。
- 顯示偵測結果。

完成標準：

- 單張照片中可穩定抓出黑色拼圖片輪廓。

### 16.2 MVP 1：單片拼圖匹配

目標：

辨識單片拼圖對應哪個 slot。

任務：

- 建立 board config。
- 建立 slot masks。
- 偵測單片拼圖片。
- 實作面積篩選。
- 實作 `matchShapes` 排序。
- 實作旋轉 IoU。
- 輸出最佳 slot ID。

完成標準：

- 在固定拍攝環境下，單片拼圖匹配準確率達到可接受水準。

### 16.3 MVP 2：多片拼圖匹配

目標：

同一張照片中支援多片拼圖。

任務：

- 偵測多個 piece contour。
- 每片 piece 產生候選 slot。
- 處理重複 slot。
- 加入 Hungarian algorithm。
- 輸出多片結果。

完成標準：

- 多片彼此不接觸時，可穩定輸出各自 slot ID。

### 16.4 MVP 3：標註與管理工具

目標：

提升建立底板資料的效率。

任務：

- 建立簡單標註工具。
- 支援新增、刪除、修改 slot polygon。
- 自動產生 slot mask。
- 儲存 board config。
- 顯示 slot ID overlay。

完成標準：

- 使用者可不用直接編輯 JSON，就能建立底板資料。

### 16.5 MVP 4：API 與 UI

目標：

提供可整合的辨識服務。

任務：

- FastAPI endpoint。
- 上傳圖片。
- 選擇 board。
- 回傳 JSON。
- 顯示 debug overlay。

完成標準：

- 可透過 API 或 Web UI 完成辨識流程。

## 17. 測試策略

### 17.1 單元測試

測試項目：

- 四角排序是否正確。
- perspective transform 是否輸出指定尺寸。
- polygon 轉 mask 是否正確。
- area、bbox、center 計算是否正確。
- IoU 計算是否正確。
- candidate filtering 是否符合預期。

### 17.2 整合測試

測試項目：

- 空底板建立流程。
- 單片拼圖辨識流程。
- 多片拼圖辨識流程。
- API 上傳與回傳格式。

### 17.3 實拍測試資料

建議建立測試集：

```text
dataset/
  board_001/
    empty/
      empty_001.jpg
    single_piece/
      slot_001_angle_000.jpg
      slot_001_angle_090.jpg
      slot_002_angle_000.jpg
    multi_piece/
      sample_001.jpg
      sample_002.jpg
```

每張測試照片需有 ground truth：

```json
{
  "image": "slot_001_angle_090.jpg",
  "pieces": [
    {
      "expected_slot_id": "slot_001",
      "expected_rotation": 90
    }
  ]
}
```

### 17.4 指標

建議追蹤：

- slot top-1 accuracy
- slot top-3 accuracy
- average IoU
- ambiguous rate
- rejected rate
- false positive piece detection count
- processing time per image

## 18. 風險與對策

| 風險 | 影響 | 對策 |
| --- | --- | --- |
| 拍攝角度變化大 | 形狀變形，匹配失敗 | 使用定位點與透視校正 |
| 光影造成黑色誤判 | 偵測到陰影或漏掉拼圖 | 均勻光源、HSV threshold、morphology |
| 黑色拼圖反光 | 邊緣破碎 | 調整光源角度，使用偏振片或霧面材質 |
| 多片拼圖接觸 | contour 合併 | MVP 規定不可接觸，後期加 watershed |
| slot 形狀很相似 | 容易判錯 | 使用 IoU、第二名差距、人工確認 |
| 底板模板不準 | 全部匹配結果受影響 | 標註工具加入 overlay 檢查 |
| 解析度不足 | 輪廓細節遺失 | 提高拍照解析度與校正後尺寸 |
| 拼圖片可能翻面 | 形狀鏡像 | 加入 mirror matching |

## 19. 開發優先順序

建議按以下順序執行：

```text
1. 收集實拍照片
2. 寫黑色拼圖 threshold 實驗
3. 建立底板透視校正
4. 手動建立少量 slot mask
5. 實作單片拼圖匹配
6. 加入 debug overlay
7. 用實拍照片調整 threshold 與 IoU 門檻
8. 擴充到完整底板 slot
9. 支援多片辨識
10. 包成 API 與 UI
```

## 20. 第一版實作規格

第一版只做最小可行系統。

### 20.1 支援功能

- 載入一張校正後的底板圖。
- 載入已建立的 slot masks。
- 載入一張包含單片黑色拼圖的照片。
- 偵測黑色拼圖 contour。
- 對所有 slot 做形狀匹配。
- 回傳最佳 slot ID。
- 輸出 debug 圖。

### 20.2 暫不支援

- 多片接觸切割。
- 自動 slot 標註。
- 深度學習模型。
- 多底板自動識別。
- 使用者帳號系統。
- 複雜前端。

### 20.3 第一版命令列介面

```bash
python scripts/recognize_image.py \
  --board-id board_001 \
  --image data/samples/input_001.jpg \
  --debug
```

輸出：

```json
{
  "board_id": "board_001",
  "pieces": [
    {
      "piece_id": "piece_001",
      "matched_slot_id": "slot_017",
      "rotation": 92,
      "confidence": 0.91,
      "status": "confident"
    }
  ]
}
```

## 21. 後續決策點

開發過程中需要盡早確認以下問題：

- 底板是否能貼定位點？
- 拼圖片是否可能翻面？
- 實際照片中拼圖片是否會互相接觸？
- 每片拼圖大小差異是否明顯？
- 底板 slot 的形狀是否有高度相似的情況？
- 需要即時辨識，還是可以接受數秒處理時間？
- 使用場景是手機拍照、固定攝影機，還是工業相機？

這些答案會影響後續是否要加入鏡像匹配、全域最佳化、標註工具或 AI segmentation。

## 22. 建議的下一步

下一步建議先完成實驗資料收集：

1. 拍攝一張空底板照片。
2. 拍攝 5 至 10 張單片黑色拼圖照片。
3. 每張照片記錄正確 slot ID。
4. 確認是否能穩定 threshold 出黑色拼圖。
5. 建立 3 至 5 個 slot mask 做初步匹配測試。

只要這個小範圍測試成功，就可以擴大到完整底板與多片辨識。
