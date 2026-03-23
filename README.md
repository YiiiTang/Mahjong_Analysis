# 麻將聽牌預測與牌譜分析系統 (Mahjong Tenpai Prediction & Analysis)

[cite_start]本專案是一個基於 Python 的麻將牌譜分析工具，結合了向聽數計算與機器學習模型（線性回歸與邏輯回歸），旨在透過玩家的打牌行為預測其是否已經進入「聽牌」狀態 。

## 🗂️ 檔案結構與說明

### 1. 核心邏輯與解析
* **`fileProcess.py`**: 定義基礎資料結構，包含 `playerState`（紀錄手牌、向聽數）及 `RoundState`（紀錄每一巡的狀態）。
* **`analyze.py`**: 負責解析原始牌譜檔案（UTF-16-LE），模擬摸打過程並更新玩家手牌。
* **`mjanalyzer_local.py`**: 核心計算模組，負責計算向聽數 (Shanten) 並提供打牌與副露建議。

### 2. 特徵工程與資料提取
* **`find_rule_all.py`**: 掃描資料夾中的牌譜，針對所有玩家生成快照，計算 3-7 中張比例、丟字比例、花色分散度等特徵。
* **`find_rule_winner.py`**: 專注於提取該局「最終獲勝者」的打牌數據進行特徵分析。

### 3. 機器學習模型訓練
* **線性回歸 (Linear Regression)**:
    * `train_model_linear_all.py` / `winner.py`: 針對數據進行線性回歸訓練並輸出特徵權重。
* **邏輯回歸 (Logistic Regression)**:
    * `train_model_logistic_all.py` / `winner.py`: 使用二元分類預測玩家是否已聽牌 (Y/N)。

---

## 🚀 快速開始

### 1. 環境準備
請確保已安裝必要的 Python 函式庫：
```bash
pip install pandas numpy scikit-learn openpyxl