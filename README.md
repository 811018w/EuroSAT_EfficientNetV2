# EuroSAT 衛星影像分類專案
  (由於訓練出來的model過大無法放入github 故放在google drive中)

  model: best_effv2_m.pt

  google drive: https://drive.google.com/drive/folders/14ImjT7FN3tl2OCSoveXeDnyiehO6kQ24?usp=sharing

  M_0.8_0.1_0.1.zip (檔案放在壓縮檔內)
  
  檔案名稱解釋:
  
    m-> 中型模型 
    
    0.8->train data ratio 
    
    first 0.1->validation data ratio
    
    second 0.1->test data ratio 

## 專案簡介

本專案利用 PyTorch 搭配 EfficientNetV2 模型，對 EuroSAT 衛星影像資料集進行分類。目標是辨識不同的地表覆蓋類別（例如農作物、森林、河流等），並達到高準確率的分類效果。

---

## 環境與資料

* 使用 PyTorch 與 torchvision
* 資料集：EuroSAT，包含多類衛星影像，分成訓練、驗證、測試集
* 輸入影像經過 Resize、正規化等前處理

---

## 模型架構

* 採用 EfficientNetV2（可選擇 s、m、l 三種規模，預設為 m）
* 最後分類層調整為資料集類別數
* 可選擇是否凍結主幹網路，只微調分類頭

---

## 訓練設定

* 批次大小：64
* 最大訓練週期：64 Epochs
* 學習率：0.0005（AdamW 優化器）
* 損失函數：交叉熵
* 梯度累積可設定（預設為 1）
* 使用自動混合精度加速（AMP）
* 使用 ReduceLROnPlateau 調整學習率
* 設置 Early Stopping，驗證準確度 5 次未提升即停止

---

## 訓練流程重點

* 每 Epoch 進行訓練與驗證
* 記錄 train\_loss、val\_loss、train\_acc、val\_acc
* 儲存驗證準確度最佳模型權重

---

## 訓練結果分析

### Accuracy 曲線

* 訓練準確度（train\_acc）快速提升至 99.9%以上
* 驗證準確度（val\_acc）穩定在約 98.8%左右，無明顯過擬合現象

### Loss 曲線

* 訓練損失（train\_loss）快速下降至接近 0
* 驗證損失（val\_loss）穩定在約 0.06～0.07，曲線較平穩

（可參考附圖 `accuracy_curve.png` 與 `loss_curve.png`）

---

## 測試結果（分類報告）

使用 scikit-learn 的 classification\_report，對測試集做詳細評估，結果如下：

| 類別                   | 精確率(precision) | 召回率(recall) | F1分數(f1-score) | 支持數量(support) |
| -------------------- | -------------- | ----------- | -------------- | ------------- |
| AnnualCrop           | 0.9770         | 0.9933      | 0.9851         | 300           |
| Forest               | 0.9933         | 0.9966      | 0.9950         | 297           |
| HerbaceousVegetation | 0.9901         | 0.9901      | 0.9901         | 302           |
| Highway              | 0.9853         | 0.9963      | 0.9908         | 269           |
| Industrial           | 0.9922         | 0.9844      | 0.9883         | 257           |
| Pasture              | 0.9837         | 0.9679      | 0.9757         | 187           |
| PermanentCrop        | 0.9842         | 0.9803      | 0.9822         | 254           |
| Residential          | 0.9904         | 0.9904      | 0.9904         | 314           |
| River                | 0.9959         | 0.9837      | 0.9897         | 245           |
| SeaLake              | 0.9964         | 0.9964      | 0.9964         | 275           |

* 整體準確率達到 **98.89%**
* Macro 平均與加權平均 F1 分數皆接近 0.989，表現穩定且優異

---

## 訓練日誌摘要

部分訓練 Epoch 重要記錄如下（訓練損失/訓練準確度/驗證損失/驗證準確度）：

* Epoch 6: 0.0412 / 0.9866 / 0.0773 / 0.9774
* Epoch 10: 0.0068 / 0.9979 / 0.0663 / 0.9856
* Epoch 13: 0.0008 / 0.9998 / 0.0684 / 0.9874
* Epoch 18: 0.0002 / 1.0000 / 0.0670 / 0.9870
* Epoch 21: 0.0003 / 1.0000 / 0.0688 / 0.9885

訓練在第 21 Epoch 透過 Early Stopping 停止，取得最佳效能。

---

## 未來優化方向

* 探索其他模型變體（如 EfficientNetV2-s 或 V2-l）比較效能差異
* 嘗試資料增強技巧以提升泛化能力
* 進一步調整超參數如學習率、批次大小等
* 使用更複雜的後處理策略提升分類穩定度
