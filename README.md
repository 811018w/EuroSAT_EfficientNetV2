# README
(由於訓練出來的model過大無法放入github 故放在eeclass中繳交)
## 專案名稱

基於 EfficientNet‑V2 微調之 EuroSAT 衛星影像分類模型

## 專案簡介

本專案運用 PyTorch 框架，結合 torchvision 的 EfficientNet‑V2 模型，對 EuroSAT 衛星影像資料集進行微調（Fine-tuning），完成 10 類地表影像的分類任務。
訓練過程中使用自動混合精度（AMP）、資料增強、EarlyStopping 及動態學習率調整等策略，以達到高準確率且有效率的模型訓練。

## 環境需求

* Python 3.8 以上
* PyTorch
* torchvision
* numpy
* matplotlib
* tqdm
* scikit-learn

安裝範例：

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn
```

## 使用說明

### 1. 下載資料集

程式會自動從 torchvision.datasets 下載 EuroSAT 資料集。

### 2. 參數設定

可以在執行時透過命令列參數調整：

* `--data_dir`：資料存放路徑，預設 `./data`
* `--batch_size`：批次大小，預設 64
* `--epochs`：訓練週期數，預設 64
* `--lr`：學習率，預設 0.0005
* `--accum_steps`：梯度累積步數，預設 1
* `--variant`：EfficientNet‑V2 版本，支援 `s`、`m`、`l`，預設 `m`
* `--freeze_backbone`：是否凍結骨幹網路，只訓練分類層（分類頭）

### 3. 執行訓練

```bash
python dl3.py --epochs 20 --batch_size 64 --variant m
```

### 4. 模型與結果

* 訓練過程中會輸出訓練及驗證的 Loss、Accuracy。
* 使用 ReduceLROnPlateau 調整學習率，並有 EarlyStopping 防止過擬合。
* 最佳模型權重儲存於 `best_effv2_[variant].pt`。
* 會產生並顯示 Loss 曲線與 Accuracy 曲線圖。

### 5. 測試與報告

程式最後載入最佳模型於測試集上進行評估，並輸出分類報告（包含 Precision、Recall、F1-score）。

---

## 實驗結果

* 模型最高驗證準確率達約 98.8%。
* 各類別 Precision、Recall、F1-score 平均皆達 0.98 以上。
* 訓練曲線與驗證曲線顯示穩定收斂，訓練過程無明顯過擬合。
* 使用自動混合精度與 AdamW 優化器有效提升訓練效率與準確度。

### 附圖

* Accuracy 曲線 (accuracy\_curve.png)
* Loss 曲線 (loss\_curve.png)

（可見附件中）

---

## 程式架構說明

* `參數設定`：使用 argparse 方便命令列調整。
* `資料處理`：使用 torchvision.transforms 進行資料增強與正規化。
* `模型定義`：調用 torchvision 內建 EfficientNetV2 模型並替換最後分類層。
* `訓練迴圈`：包含梯度累積、混合精度、EarlyStopping、動態學習率。
* `評估`：驗證集和測試集評估，使用 sklearn 產出詳細分類報告。
* `繪圖`：matplotlib 繪製訓練過程中的準確率與損失曲線。

---

## 未來改進方向

* 可嘗試多光譜影像輸入或自注意力機制。
* 增加資料集擴增策略，提升模型泛化能力。
* 部署於邊緣裝置以驗證實際推論效能。
