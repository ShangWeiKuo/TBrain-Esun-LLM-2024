import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Retrieval:
    def __init__(self, tokenizer_name, model_name):
        """初始化 tokenizer 和模型。

        Args:
            tokenizer_name (str): tokenizer 名稱
            model_name (str): model 名稱
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def langchain_rerank_retrieve(self, query, source_files, documents, chunk_size):
        """使用 langchain rerank 演算法進行文件檢索。

        Args:
            query (str): 使用者輸入的查詢字串。
            source_files (list): 所有文件。
            documents (list): 所有文件的內容。
            chunk_size (int): 每個文件的切分大小。

        Returns:
            list: 經過 rerank 後的文件 id 清單。
        """

        filtered_corpus = []
        chunk_file_map = []

        for file in source_files:
            for idx in range(len(documents)):
                if int(file) == documents[idx].metadata["pid"]:
                    chunks = [
                        documents[idx].page_content[i : i + chunk_size]
                        for i in range(0, len(documents[idx].page_content), chunk_size)
                    ]
                    filtered_corpus.extend(chunks)
                    chunk_file_map.extend([file] * len(chunks))

        pairs = [[query, filtered_corpus[i]] for i in range(len(filtered_corpus))]

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        top_file_id = chunk_file_map[scores.argmax()]

        return top_file_id

    # 檢索流程主程式
    def process_retrieval_rerank(self, questions, corpora, chunk_size=512):
        """處理一組問題的檢索，並保存結果。

        Args:
            questions (list): 問題列表。
            corpora (dict): 文檔字典。
            chunk_size (int): 文檔塊的大小。

        Returns:
            list: 處理後的結果列表。
        """

        results_rerank = []
        for q_dict in tqdm(questions):
            category = q_dict["category"]
            query = q_dict["query"]
            source = q_dict["source"]

            if category in corpora:
                retrieved_rerank = self.langchain_rerank_retrieve(
                    query, source, corpora[category], chunk_size
                )
                results_rerank.append(retrieved_rerank)
            else:
                raise ValueError(f"未知的類別：{category}")

        return results_rerank
