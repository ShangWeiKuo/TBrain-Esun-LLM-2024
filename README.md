# TBrain-Esun-LLM-2024

## 隊伍名稱：TEAM_6669

<br>

## 資源配置
* 硬體：NVIDIA GeForce RTX 4090 顯卡
* 軟體：Windows 11 作業系統、Python 3.10.15、PyTorch 2.5.0、CUDA 11.8、CUDNN 8.6.0、NVIDIA Driver 516.94

<br>

## 環境建置與執行

本專案是以 Anaconda 建立虛擬環境，並使用 PyTorch 框架進行預測，請參考以下步驟進行環境建置與執行。

### 1. 環境建立
```
conda create --name [env_name] python=3.10.15 -y
```
### 2. config 設定
```
(1) 到 data_path.ini 設定資料路徑
    [Data]
    source_path = reference                                         # PDF 原始資料來源路徑
    questions_path = dataset/preliminary/questions_preliminary.json # 問題資料來源路徑
    output_path = dataset/preliminary/pred_retrieve.json            # pred_retrieve 檔案輸出路徑
    insurance_json_name = insurance_data.json                       # insurance 文本資料來源路徑
    finance_json_name = finance_data.json                           # Finance 文本資料來源路徑
    faq_json_name = pid_map_content.json                            # FAQ 文本資料來源路徑

(2) 到 model_name.ini 設定模型參數
    [Model]
    tokenizer_name = BAAI/bge-reranker-v2-m3                        # Tokenizer 名稱
    model_name = BAAI/bge-reranker-v2-m3                            # 模型名稱
```
### 3. 環境啟動並執行程式
```
conda activate [env_name]
python main.py
```
### 4. 執行結果
在使用者指定的 output_path  路徑下，會產生 pred_retrieve.json 檔案。