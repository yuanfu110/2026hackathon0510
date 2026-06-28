# 2026 Hackathon 0510

這是一個以 Flask 和 Ultralytics YOLO 建立的影像辨識專案，用於辨識回收/垃圾分類影像，並提供網頁介面、API、即時狀態輸出與資料收集輔助工具。

專案依照開發進度保留三個版本：

- `v0`: 基礎 YOLO 影像辨識 API 與網頁介面。
- `v1`: 加入 `sample.json` 狀態輸出、測試/資料收集伺服器、訓練輔助腳本與資料集骨架。
- `v2`: 加入顯示端頁面、圖片請求 API、Server-Sent Events 狀態串流、最新圖片標註流程與更新後的模型權重。

## 專案結構

```text
.
├── v0/
│   ├── app.py
│   ├── test_server.py
│   ├── TRAINING.md
│   ├── v0.pt
│   ├── best_model.pt
│   └── yolov8n.pt
├── v1/
│   ├── app.py
│   ├── test_server.py
│   ├── sample.json
│   ├── templates/
│   ├── scripts/
│   ├── dataset/trash/
│   └── models/
└── v2/
    ├── app.py
    ├── test_server.py
    ├── sample.json
    ├── templates/
    ├── scripts/
    ├── dataset/trash/
    ├── models/
    └── *.pt
```

## 環境需求

- Python 3.10 或更新版本
- Flask
- Ultralytics
- Pillow

目前專案沒有獨立的 `requirements.txt`，可先用以下方式安裝：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install flask ultralytics pillow
```

## 執行最新版

建議優先使用 `v2`：

```powershell
cd v2
python app.py
```

啟動後開啟：

```text
http://localhost:5000
```

`v2/app.py` 預設載入 `v3.pt`。如需改用其他模型，可設定 `MODEL_PATH`：

```powershell
$env:MODEL_PATH="models/trash_best.pt"
python app.py
```

## 執行資料收集/測試伺服器

每個版本都有 `test_server.py`，可用來收集或標註影像資料：

```powershell
cd v2
python test_server.py
```

此伺服器同樣使用 `http://localhost:5000`，因此請先停止正在執行的 `app.py`。

## 訓練資料與模型

訓練說明放在各版本的 `TRAINING.md`。`v1` 和 `v2` 保留了 `dataset/trash/` 的資料集骨架與訓練腳本：

```powershell
cd v2
python scripts/check_dataset.py
python scripts/train_trash.py
```

實際訓練圖片與標註檔請放在 `dataset/trash/images/` 和 `dataset/trash/labels/` 對應 split 中。

## Git 追蹤範圍

已納入版本控制的內容包含：

- 各版本的 Flask 程式碼
- HTML templates
- 訓練/檢查腳本
- 空資料集結構
- 必要模型權重
- `sample.json` 範例狀態檔

以下內容刻意不納入 git：

- `.venv/`: 本機 Python 虛擬環境
- `uploads/`: 使用者上傳或暫存圖片
- `runs/`: YOLO 訓練輸出
- `dataset/capture/`: 實拍收集資料
- `__pycache__/`: Python cache
- `best.pt`: 目前是 52 byte 文字檔，不是有效 YOLO checkpoint

如果需要保存大量訓練影像或輸出模型，建議另外使用雲端硬碟、GitHub Releases 或模型/資料集儲存服務。
