# 麻將聽牌預測與牌譜分析系統 (Mahjong Tenpai Prediction & Analysis)

本專案是一個基於 Python 的麻將牌譜分析工具，結合了向聽數計算與機器學習模型（線性回歸與邏輯回歸），旨在透過玩家的打牌行為預測其是否已經進入「聽牌」狀態。

## 🗂️ 檔案結構與說明

### 1. 核心邏輯與解析
* **`fileProcess.py`**: 定義基礎資料結構，包含 `playerState`（紀錄手牌、向聽數）及 `RoundState`（紀錄每一巡的狀態）。
* **`analyze.py`**: 負責解析原始牌譜檔案（UTF-16-LE），模擬摸打過程並更新玩家手牌。
* **`mjanalyzer_local.py`**: 核心計算模組，負責計算向聽數 (Shanten) 並提供打牌與副露（吃、碰、槓）建議。

### 2. 特徵工程與資料提取
* **`find_rule_all.py`**: 掃描資料夾中的牌譜，針對所有玩家生成快照，計算 3-7 中張比例、丟字比例、花色分散度等特徵，並匯出為 `analyze_all_players_snapshots.xlsx`。
* **測試集資料提取**: 針對測試集資料夾（Test），萃取相同的特徵並標註實際聽牌狀態，匯出為 `analyze_test_snapshots.xlsx` 以供驗證。

### 3. 機器學習模型訓練與評估 (`Training_Model/`)
本專案的機器學習模型訓練與特徵評估腳本均放置於此資料夾，包含自動權重更新與消融實驗（Ablation Studies）：
* **自動訓練與權重注入 (Auto-Training)**:
    * **線性回歸訓練**: 讀取快照資料針對 0-4 副露分別訓練模型。腳本會自動利用正規表達式，將算出的權重、平均值、標準差與截距，跨資料夾直接寫入並更新 `track_tenpai_state_linear.py` 的參數區塊。
    * **邏輯回歸訓練**: 讀取資料進行訓練，並計算預測準確率，訓練完成後同樣會自動將最新權重注入 `track_tenpai_state_logistic.py` 腳本中。
* **特徵消融實驗 (Ablation Studies)**:
    * 透過逐一移除單一特徵（如：移除特定花色集中度或丟牌行為）並計算「模型準確率下降幅度」，來量化個別特徵對聽牌預測的影響力。
    * 腳本會自動針對線性回歸與邏輯回歸，輸出各副露狀態下的特徵重要性水平長條圖，方便視覺化評估。

### 4. 機器學習模型追蹤與預測
本系統包含使用預先訓練好的權重（針對不同副露數量有不同的權重、平均值與標準差）來進行逐巡追蹤預測的腳本：
* **線性回歸追蹤 (`track_tenpai_state_linear.py`)**: 使用 `WEIGHTS_BY_MELD` 計算特徵的線性分數，藉此評估聽牌進度，產出 `Batch_Linear_Tracking_Result.xlsx` 與 JSON 紀錄檔。
* **邏輯回歸追蹤 (`track_tenpai_state_logistic.py`)**: 使用 Sigmoid 函數將特徵轉換為 0 到 1 之間的「預測聽牌機率」，產出 `Batch_Logistic_Tracking_Result.xlsx` 與 JSON 紀錄檔。

### 5. 🕵️ 模型預測倒退 (Flickering) 診斷工具
本專案提供針對模型預測穩定性的診斷腳本，專門找出「上一巡判斷已聽牌，下一巡卻因某些打牌動作而改判未聽牌」的閃爍/倒退 (Flickering) 案例：
* **線性回歸倒退分析 (`find_flickering_cases_linear.py`)*: 讀取線性回歸的追蹤結果，找出預測分數跌破閾值 (0.5) 的盤面，輸出 `Flickering_Cases_Linear.xlsx`[cite: 17]。
* **邏輯回歸倒退分析 (`find_flickering_cases_logistic.py`)*: 讀取邏輯回歸的追蹤結果，找出預測機率跌破閾值 (0.5) 的盤面，輸出 `Flickering_Cases_Logistic.xlsx`[cite: 18]。
* **雙模型共同倒退交叉比對 (`find_common_flickering.py`)*: 交叉比對上述兩份報告，找出「兩個模型在同一巡共同發生判斷倒退」的嚴重案例，並計算出綜合跌幅，最終匯出 `Common_Flickering_Cases.xlsx` 供開發者進行進階的盤面覆盤與模型除錯[cite: 16]。

### 6. 📊 視覺化與分析工具
專案內建強大的 matplotlib 視覺化模組，用於觀察特徵變化與聽牌趨勢：
* **多局平均趨勢圖**: 讀取追蹤結果（如 `Batch_Linear_Tracking_Result.xlsx`），計算各巡數的特徵平均值與變異數（標準差陰影），並繪製包含所有特徵與預測分數的綜合趨勢圖。
* **單局特徵批次調查 (GUI)**: 提供基於 `tkinter` 的圖形化介面，讓使用者選擇牌譜資料夾，程式會自動為每一局的各個玩家（東、南、西、北）繪製獨立的特徵與實際向聽數變化圖表，並特別標註該局的獲勝者。

---

## 🚀 快速開始

### 1. 環境準備
請確保您的環境已安裝 Python 3.x，並透過專案內附的 `requirements.txt` 快速安裝所有依賴套件：
```bash
pip install -r requirements.txt
```
> 套件包含 `pandas`, `numpy`, `scikit-learn` 與 `openpyxl`。（視覺化相關圖表會呼叫內建的 `tkinter` 與 `matplotlib`）

### 2. 執行流程範例

#### 步驟一：解析牌譜與特徵提取
將原始的麻將牌譜檔案放置於指定目錄。執行特徵提取腳本，將資料轉化為可供模型使用的快照格式：
```bash
python find_rule_all.py
```

#### 步驟二：模型訓練與參數更新 (選用)
進入 `Training_Model` 資料夾，執行自動訓練腳本，程式將自動把最新算出的權重參數寫入根目錄的預測腳本中：
```bash
cd Training_Model
python train_model_linear_all.py
```

#### 步驟三：模型預測與追蹤
選擇使用線性回歸或邏輯回歸腳本，對測試集的牌局進行逐巡掃描，預測每位玩家的聽牌狀態：
```bash
# 回到專案根目錄執行
python track_tenpai_state_logistic.py
```

#### 步驟四：模型穩定性診斷 (選用)
如果需要覆盤模型發生「預測倒退」的極端盤面，可以依序執行以下腳本：
```bash
python find_flickering_cases_linear.py
python find_flickering_cases_logistic.py
python find_common_flickering.py
```
> 執行後會產生彙整報表，列出綜合跌幅最嚴重的共同倒退案例，並在終端機中印出 Top 5 預覽[cite: 16, 17, 18]。

#### 步驟五：資料視覺化
要觀察模型預測分數與特徵在不同巡數的變化，可執行繪圖腳本：
```bash
# 產出多局平均趨勢圖與子圖 (儲存於 Plot_Results 資料夾)
python plot_multigame_averages.py

# 啟動 GUI 進行單局特定玩家的特徵變化調查
python plot_game_explorer.py
```

---

## ⚠️ 重要注意事項

1. **牌譜編碼格式**：原始牌譜 `.txt` 檔案必須為 **UTF-16-LE** 編碼格式，否則 `analyze.py` 在讀取時會發生錯誤。
2. **自動更新權重標籤**：`track_tenpai_state_linear.py` 與 `track_tenpai_state_logistic.py` 內部設有 `# AUTO-UPDATE-START` 與 `# AUTO-UPDATE-END` 註解標籤。**請勿刪除或修改這些註解**，否則訓練腳本將無法自動注入更新後的模型權重。