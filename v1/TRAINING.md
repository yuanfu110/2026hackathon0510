# Trash Model Training

## Folder layout

Put training images and YOLO label files here:

```text
dataset/trash/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

Each image needs a matching label file with the same base name:

```text
dataset/trash/images/train/example.jpg
dataset/trash/labels/train/example.txt
```

YOLO label format:

```text
class_id x_center y_center width height
```

All box values must be normalized from 0 to 1.

## Classes

```text
0 一般垃圾
1 塑膠
2 紙類
3 金屬
4 玻璃
5 廚餘
```

## Check the dataset

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset.py
```

## Train

```powershell
.\.venv\Scripts\python.exe scripts\train_trash.py --epochs 100 --imgsz 640 --batch 8
```

Training output will be written under:

```text
runs/trash/train/
```

The best weights will also be copied to:

```text
models/trash_best.pt
```

## Run the app with trained weights

```powershell
$env:MODEL_PATH="models/trash_best.pt"
.\.venv\Scripts\python.exe app.py
```
