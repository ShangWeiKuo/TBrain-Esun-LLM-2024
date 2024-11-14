## AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用

### 專案結構
```
.
├── Preprocess
│   ├── data_preprocess.py    # 使用 pdfplumber 從 PDF 中提取文字，並進行文字轉換與 markdown 切分
│   └── README.md             # 預處理步驟的文件說明
├── Model
│   ├── retrieval.py          # 使用 langchain rerank 演算法進行文件檢索
│   └── README.md             # 檢索步驟的文件說明
├── config
│   ├── data_path.ini         # 資料路徑設定檔
│   ├── data_path.py          # 讀取 data_path.ini 檔案的方法
│   ├── model_path.ini        # 模型路徑設定檔
│   └── model_path.py         # 讀取 model_path.ini 檔案的方法
├── dataset                   # 所需的數據集
│   ├── preliminary           # 初賽需預測檔案存放資料夾 (手動下載後放至此資料夾)
│   └── reference             # 初賽 PDF 檔存放資料夾 (手動下載後解壓縮至此資料夾)
├── main.py                   # 執行預處理和檢索的主程式
├── requirements.txt          # 所需的 Python 套件
└── README.md                 # 專案說明文件
```

---

### 資源配置
* 硬體：NVIDIA GeForce RTX 4090 顯卡
* 軟體：
    * Windows 11 作業系統
    * Python 3.10.15
    * PyTorch 2.5.0
    * CUDA 11.8
    * CUDNN 8.6.0
    * NVIDIA Driver 516.94
    * Anaconda 22.9.0

---

### 環境建置與執行
本專案是以 Anaconda (version 22.9.0) 建立虛擬環境，並使用 PyTorch 框架進行預測，請參考以下步驟進行環境建置與執行。

#### 1. 建立 Anaconda 環境
在命令列執行以下指令來建立虛擬環境：
```
conda create --name [env_name] python=3.10.15 -y
```
#### 2. 配置檔案設定
在執行專案之前，請確認以下配置：

(1) 設定資料路徑
編輯 `data_path.ini` 檔案，設定各種數據路徑：
```ini
[Data]
source_path = dataset/reference                                 # 初賽 PDF 檔存放資料夾路徑 (手動下載後解壓縮至此資料夾)
questions_path = dataset/preliminary/questions_preliminary.json # 問題資料來源路徑
output_path = dataset/preliminary/pred_retrieve.json            # pred_retrieve 檔案輸出路徑
insurance_json_name = insurance_data.json                       # insurance 文本資料保存檔名
finance_json_name = finance_data.json                           # Finance 文本資料保存檔名
faq_json_name = pid_map_content.json                            # FAQ 文本資料保存檔名
```
(2) 設定模型參數
編輯 `model_name.ini` 檔案，設定模型參數：
```ini
[Model]
tokenizer_name = BAAI/bge-reranker-v2-m3                        # Tokenizer 名稱
model_name = BAAI/bge-reranker-v2-m3                            # 模型名稱
```
#### 3. 啟動環境並執行主程式
在命令列執行以下指令來啟動環境並執行主程式：
```
conda activate [env_name]
python main.py
```
#### 4. 檢視結果
主程式執行完成後，會在使用者指定的 `output_path` 路徑下產生 `pred_retrieve.json` 檔案，內容為最終檢索結果。

---

### Huggingface transformers （bge-reranker-v2-m3）模型使用說明
以下是如何在程式中使用 `bge-reranker-v2-m3` 模型的範例程式碼：
```
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

# [[問題, chunk 後的文本], [...], ...]
pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```