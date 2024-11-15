import json
from config.data_path import (
    source_path,
    questions_path,
    output_path,
    insurance_json_name,
    finance_json_name,
    faq_json_name,
)
from config.model_name import tokenizer_name, model_name
from Preprocess.data_preprocess import PreProcess
from Model.retrieval import Retrieval


def output_pre_retrieval(output_path, questions, results_rerank):
    """產出提交檔案

    Args:
        questions (dict): 比賽問題
        results_rerank (list): 檢索結果
    """
    answer_dict = {"answers": []}  # 初始化字典

    for idx, q_dict in enumerate(questions):
        # 將結果加入字典
        answer_dict["answers"].append(
            {"qid": q_dict["qid"], "retrieve": results_rerank[idx]}
        )

    # 將答案字典保存為json文件
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(
            answer_dict, f, ensure_ascii=False, indent=4
        )  # 儲存檔案，確保格式和非ASCII字符


def main():
    """主程式"""
    # 初始化數據預處理方法
    data_preprocess = PreProcess(
        source_path,
        questions_path,
        insurance_json_name,
        finance_json_name,
        faq_json_name,
    )
    # 初始化模型
    retrieval = Retrieval(tokenizer_name, model_name)

    corpora, questions = data_preprocess.get_raw_data()

    # 處理 finance 的日期（轉換中文數字成阿拉伯數字）
    corpora["finance"] = data_preprocess.process_dict(corpora["finance"])
    doc_dict = data_preprocess.build_documents(corpora)

    results_rerank = retrieval.process_retrieval_rerank(
        questions, doc_dict, chunk_size=512
    )

    output_pre_retrieval(output_path, questions, results_rerank)


if __name__ == "__main__":
    main()
