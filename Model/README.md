# Retrieval 方法

retrieval.py 用於文件檢索，以下檢索時用到的函數：

1. <b>langchain_rerank_retrieve</b>：
使用 langchain rerank 演算法進行文件檢索。

2. <b>process_retrieval_rerank</b>：
檢索流程主程式，負責執行 langchain_rerank_retrieve 函數。